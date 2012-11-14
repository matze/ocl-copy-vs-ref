#include <CL/cl.h>
#include <stdio.h>
#include "ocl.h"

static const gchar* opencl_error_msgs[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "CL_MISALIGNED_SUB_BUFFER_OFFSET",
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",

    /* next IDs start at 30! */
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE"
};

const gchar* opencl_map_error(int error)
{
    if (error >= -14)
        return opencl_error_msgs[-error];
    if (error <= -30)
        return opencl_error_msgs[-error-15];
    return NULL;
}

gchar *
ocl_read_program(const gchar *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
        return NULL;

    fseek(fp, 0, SEEK_END);
    const size_t length = ftell(fp);
    rewind(fp);

    gchar *buffer = (gchar *) g_malloc0(length+1);
    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }

    size_t buffer_length = fread(buffer, 1, length, fp);
    fclose(fp);
    if (buffer_length != length) {
        g_free(buffer);
        return NULL;
    }
    return buffer;
}

cl_program
ocl_get_program(opencl_desc *ocl, const gchar *filename, const gchar *options)
{
    gchar *buffer = ocl_read_program(filename);
    if (buffer == NULL)
        return FALSE;

    int errcode = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ocl->context, 1, (const char **) &buffer, NULL, &errcode);

    if (errcode != CL_SUCCESS) {
        g_free(buffer);
        return NULL;
    }

    errcode = clBuildProgram(program, ocl->num_devices, ocl->devices, options, NULL, NULL);

    if (errcode != CL_SUCCESS) {
        const int LOG_SIZE = 4096;
        gchar* log = (gchar *) g_malloc0(LOG_SIZE * sizeof(char));
        CHECK_ERROR(clGetProgramBuildInfo(program, ocl->devices[0], CL_PROGRAM_BUILD_LOG, LOG_SIZE, (void*) log, NULL));
        g_print("\n=== Build log for %s===%s\n\n", filename, log);
        g_free(log);
        g_free(buffer);
        return NULL;
    }

    g_free(buffer);
    return program;
}

static cl_platform_id
get_nvidia_platform(void)
{
    cl_platform_id *platforms = NULL;
    cl_uint num_platforms = 0;
    cl_platform_id nvidia_platform = NULL;

    cl_int errcode = clGetPlatformIDs(0, NULL, &num_platforms);
    if (errcode != CL_SUCCESS)
        return NULL;

    platforms = (cl_platform_id *) g_malloc0(num_platforms * sizeof(cl_platform_id));
    errcode = clGetPlatformIDs(num_platforms, platforms, NULL);

    gchar result[256];

    for (int i = 0; i < num_platforms; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 256, result, NULL);
        if (g_strstr_len(result, -1, "NVIDIA") != NULL) {
            nvidia_platform = platforms[i];
            break;
        }
    }

    g_free(platforms);
    return nvidia_platform;
}

opencl_desc *
ocl_new (gboolean enable_profiling)
{
    opencl_desc *ocl = g_malloc0(sizeof(opencl_desc));

    cl_platform_id platform = get_nvidia_platform();
    if (platform == NULL)
        return NULL;

    int errcode = CL_SUCCESS;

    CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ocl->num_devices));
    ocl->devices = g_malloc0(ocl->num_devices * sizeof(cl_device_id));
    CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, ocl->num_devices, ocl->devices, NULL));

    ocl->context = clCreateContext(NULL, ocl->num_devices, ocl->devices, NULL, NULL, &errcode);
    CHECK_ERROR(errcode);

    ocl->cmd_queues = g_malloc0(ocl->num_devices * sizeof(cl_command_queue));
    cl_command_queue_properties queue_properties = enable_profiling ? CL_QUEUE_PROFILING_ENABLE : 0;

    const size_t len = 256;
    char string_buffer[len];

    CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, len, string_buffer, NULL));
    printf("# Platform: %s\n", string_buffer);

    for (int i = 0; i < ocl->num_devices; i++) {
        CHECK_ERROR(clGetDeviceInfo(ocl->devices[i], CL_DEVICE_NAME, len, string_buffer, NULL));
        printf("# Device %i: %s\n", i, string_buffer);
        ocl->cmd_queues[i] = clCreateCommandQueue(ocl->context, ocl->devices[i], queue_properties, &errcode);
        CHECK_ERROR(errcode);
    }
    return ocl;
}

void
ocl_free(opencl_desc *ocl)
{
    for (int i = 0; i < ocl->num_devices; i++)
        clReleaseCommandQueue(ocl->cmd_queues[i]);

    CHECK_ERROR(clReleaseContext(ocl->context));

    g_free(ocl->devices);
    g_free(ocl->cmd_queues);
    g_free(ocl);
}

void
ocl_show_event_info(FILE *fp, const gchar *kernel, guint num_events, cl_event *events)
{
    for (int i = 0; i < num_events; i++) {
        cl_ulong param;
        cl_event event;
        cl_command_queue queue;

        event = events[i];
        clGetEventInfo (event, CL_EVENT_COMMAND_QUEUE, sizeof (cl_command_queue), &queue, NULL);

        fprintf (fp, "%s %p", kernel, queue);
        CHECK_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &param, NULL));
        fprintf (fp, " %ld", param);
        CHECK_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &param, NULL));
        fprintf (fp, " %ld", param);
        CHECK_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &param, NULL));
        fprintf (fp, " %ld", param);
        CHECK_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &param, NULL));
        fprintf (fp, " %ld\n", param);
    }
}
