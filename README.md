# copy-vs-reference

`copy-vs-reference` is a small benchmark program to check CUDA/OpenCL run-time
decisions for multi-GPU setups. It measures two cases for N data elements: 1)
Uploading data through a single buffer and letting the run-time decide how to
transfer data between GPUs. So, for each GPU the kernel is called with the
initial buffer as an input, therefore _reference_. 2) Each GPU has its own
buffer which is filled upfront by a single thread. Hence _copy_.

_Results_: Up to now it looks that's worth to explicitly setup buffers for each
GPU and do a manual copy.
