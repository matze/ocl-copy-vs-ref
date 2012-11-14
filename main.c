/* main.c as part of mgpu
 *
 * Copyright (C) 2011-2012 Matthias Vogelgesang <matthias.vogelgesang@gmail.com>
 *
 * mgpu is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * mgpu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Labyrinth; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

#include <CL/cl.h>
#include <glib.h>
#include <stdio.h>

#include "ocl.h"

typedef struct {
    gint num_images;
    guint width;
    guint height;
} Settings;

typedef struct {
    Settings *settings;
    guint num_images;
    gsize image_size;
    gfloat **host_data;
    gfloat **single_result;
    gfloat **multi_result;
    cl_mem *dev_data_in;
    cl_mem *dev_data_out;
    cl_event *events;
    cl_event *read_events;
    cl_event *write_events;
    cl_kernel *kernels;
    opencl_desc *ocl;
    size_t global_work_size[2];
} Benchmark;

typedef struct {
    Benchmark *benchmark;
    guint thread_id;
    guint data_in_index;
} ThreadLocalData;

typedef void (*BenchmarkFunc) (Benchmark *);

static Benchmark *
setup_benchmark(opencl_desc *ocl, Settings *settings)
{
    Benchmark *b;
    cl_program program;
    cl_int errcode = CL_SUCCESS;

    program = ocl_get_program(ocl, "nlm.cl", "");

    if (program == NULL) {
        g_warning ("Could not open nlm.cl");
        ocl_free (ocl);
        return NULL;
    }

    b = (Benchmark *) g_malloc0(sizeof(Benchmark));
    b->ocl = ocl;
    b->settings = settings;

    /* Create kernel for each device */
    b->kernels = g_malloc0(ocl->num_devices * sizeof(cl_kernel));

    for (int i = 0; i < ocl->num_devices; i++) {
        b->kernels[i] = clCreateKernel(program, "nlm", &errcode);
        CHECK_ERROR(errcode);
    }

    b->global_work_size[0] = b->settings->width;
    b->global_work_size[1] = b->settings->height;
    b->num_images = b->settings->num_images < 0 ? ocl->num_devices * 16 : b->settings->num_images;
    b->image_size = b->settings->width * b->settings->height * sizeof(gfloat);
    b->single_result = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->multi_result = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->host_data = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->events = (cl_event *) g_malloc0(b->num_images * sizeof(cl_event));
    b->read_events = (cl_event *) g_malloc0(b->num_images * sizeof(cl_event));
    b->write_events = (cl_event *) g_malloc0(b->num_images * sizeof(cl_event));

    g_print("# Computing <nlm> for %i images of size %ix%i\n", b->num_images, b->settings->width, b->settings->height);

    for (guint i = 0; i < b->num_images; i++) {
        b->host_data[i] = (gfloat *) g_malloc0(b->image_size);
        b->single_result[i] = (gfloat *) g_malloc0(b->image_size);
        b->multi_result[i] = (gfloat *) g_malloc0(b->image_size);

        for (guint j = 0; j < b->settings->width * b->settings->height; j++)
            b->host_data[i][j] = (gfloat) g_random_double();
    }

    b->dev_data_in = (cl_mem *) g_malloc0(ocl->num_devices * sizeof(cl_mem));
    b->dev_data_out = (cl_mem *) g_malloc0(ocl->num_devices * sizeof(cl_mem));

    for (guint i = 0; i < ocl->num_devices; i++) {
        b->dev_data_in[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, b->image_size, NULL, &errcode);
        CHECK_ERROR(errcode);
        b->dev_data_out[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, b->image_size, NULL, &errcode);
        CHECK_ERROR(errcode);
    }

    return b;
}

static void
teardown_benchmark (Benchmark *b)
{
    for (guint i = 0; i < b->num_images; i++) {
        g_free(b->host_data[i]);
        g_free(b->single_result[i]);
        g_free(b->multi_result[i]);
    }

    for (guint i = 0; i < b->ocl->num_devices; i++) {
        CHECK_ERROR(clReleaseMemObject(b->dev_data_in[i]));
        CHECK_ERROR(clReleaseMemObject(b->dev_data_out[i]));
    }

    g_free(b->host_data);
    g_free(b->single_result);
    g_free(b->multi_result);
    g_free(b->dev_data_in);
    g_free(b->dev_data_out);
    g_free(b->events);
    g_free(b->read_events);
    g_free(b->write_events);
    g_free(b);
}

static void
measure_benchmark (const gchar *prefix, BenchmarkFunc func, Benchmark *benchmark)
{
    gdouble time;
    GTimer *timer;
    
    timer = g_timer_new();
    func (benchmark);
    g_timer_stop (timer);
    time = g_timer_elapsed (timer, NULL);
    g_timer_destroy(timer);

    g_print("# %s: total = %fs\n", prefix, time);
}

static gpointer
process_shared_buffer (gpointer data)
{
    ThreadLocalData *tld = (ThreadLocalData *) data;
    Benchmark *benchmark = tld->benchmark;
    cl_mem dev_data_in;
    cl_mem dev_data_out;
    cl_kernel kernel;
    cl_event event;
    cl_command_queue cmd_queue;
    const guint idx = tld->thread_id;

    kernel = benchmark->kernels[idx];
    cmd_queue = benchmark->ocl->cmd_queues[idx];
    dev_data_in = benchmark->dev_data_in[tld->data_in_index];
    dev_data_out = benchmark->dev_data_out[idx];

    CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &dev_data_in));
    CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &dev_data_out));

    CHECK_ERROR(clEnqueueNDRangeKernel(cmd_queue, kernel,
                2, NULL, benchmark->global_work_size, NULL,
                0, NULL, &event));

    clWaitForEvents (1, &event);

    return NULL;
}

static void
execute_pass_through (Benchmark *benchmark)
{
    GThread *threads[benchmark->ocl->num_devices];
    ThreadLocalData tld[benchmark->ocl->num_devices];

    for (guint i = 0; i < benchmark->num_images; i++) {
        /* 
         * Write data into a single shared buffer. Between GPUs, the run-time
         * implicitly transfers the data.
         */
        CHECK_ERROR (clEnqueueWriteBuffer (benchmark->ocl->cmd_queues[0], benchmark->dev_data_in[0], CL_TRUE,
                     0, benchmark->image_size, benchmark->host_data[i],
                     0, NULL, NULL));

        for (guint j = 0; j < benchmark->ocl->num_devices; j++) {
            tld[j].thread_id = j;
            tld[j].benchmark = benchmark;
            tld[j].data_in_index = 0;
            threads[j] = g_thread_create (&process_shared_buffer, &tld[j], TRUE, NULL);
        }

        for (guint j = 0; j < benchmark->ocl->num_devices; j++)
            g_thread_join(threads[j]);

        /* TODO: check data */
    }
}

static void
execute_copy (Benchmark *benchmark)
{
    GThread *threads[benchmark->ocl->num_devices];
    ThreadLocalData tld[benchmark->ocl->num_devices];

    for (guint i = 0; i < benchmark->num_images; i++) {
        for (guint j = 0; j < benchmark->ocl->num_devices; j++) {
            CHECK_ERROR (clEnqueueWriteBuffer (benchmark->ocl->cmd_queues[j], benchmark->dev_data_in[j], CL_TRUE,
                         0, benchmark->image_size, benchmark->host_data[i],
                         0, NULL, NULL));

            tld[j].thread_id = j;
            tld[j].benchmark = benchmark;
            tld[j].data_in_index = j;
            threads[j] = g_thread_create (&process_shared_buffer, &tld[j], TRUE, NULL);
        }

        for (guint j = 0; j < benchmark->ocl->num_devices; j++)
            g_thread_join(threads[j]);

        /* TODO: check data */
    }
}


int
main(int argc, char *argv[])
{
    static Settings settings = {
        .num_images = -1,
        .width = 1024,
        .height = 1024,
    };

    static GOptionEntry entries[] = {
        { "num-images", 'n', 0, G_OPTION_ARG_INT, &settings.num_images, "Number of images", "N" },
        { "width", 'w', 0, G_OPTION_ARG_INT, &settings.width, "Width of imags", "W" },
        { "height", 'h', 0, G_OPTION_ARG_INT, &settings.height, "Height of images", "H" },
        { NULL }
    };

    GOptionContext *context;
    opencl_desc *ocl;
    Benchmark *benchmark;
    GError *error = NULL;

    context = g_option_context_new (" - test multi GPU performance");
    g_option_context_add_main_entries (context, entries, NULL);

    if (!g_option_context_parse (context, &argc, &argv, &error)) {
        g_print ("Option parsing failed: %s\n", error->message);
        return 1;
    }

    g_print("## %s@%s\n", g_get_user_name(), g_get_host_name());

    g_thread_init (NULL);

    ocl = ocl_new (FALSE);
    benchmark = setup_benchmark (ocl, &settings);

    measure_benchmark ("Copy", execute_copy, benchmark);
    measure_benchmark ("Pass-through", execute_pass_through, benchmark);

    teardown_benchmark (benchmark);
    ocl_free(ocl);
    return 0;
}
