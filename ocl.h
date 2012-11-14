#ifndef OCL_H
#define OCL_H

#include <glib.h>

typedef struct {
    cl_context context;
    cl_uint num_devices;
    cl_device_id *devices;
    cl_command_queue *cmd_queues;
} opencl_desc;

opencl_desc *   ocl_new             (gboolean enable_profiling);
void            ocl_free            (opencl_desc *ocl);
gchar *         ocl_read_program    (const gchar *filename);
cl_program      ocl_get_program     (opencl_desc *ocl, const gchar *filename, const gchar *options);
void            ocl_show_event_info (FILE *fp, const gchar *kernel, guint num_events, cl_event *events);
const gchar*    opencl_map_error    (int error);

#define CHECK_ERROR(error) { \
    if ((error) != CL_SUCCESS) g_message("OpenCL error <%s:%i>: %s", __FILE__, __LINE__, opencl_map_error((error))); }

#endif
