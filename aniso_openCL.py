#pragma OPENCL EXTENSION cl_khr_fp64 : enable

from __future__ import division
import pyopencl as cl
import anisodiff as aniso
import numpy as np
import pylab
import os.path

from timer import Timer
from scipy.ndimage import imread

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

'''
We iterate over the four parallel methods with different local size width. we have verified that the first 8 digits 
of their results are consistent with the results produced by the serial version.
'''
'''
@param image_path : path of the image
@param npzfile : a boolean indicating whether the image is an npz file.
@param num_iters : nunmber of iterations
'''
def run_parallel_program(image_path, npzfile, num_iters):
    size1 = [8, 16, 32, 64, 128]
    labels = ["Block Parallel", "Column Parallel", "Column Reused Buffer Parallel", "No Buffer Parallel"]
    host_image_filtered = None
    for method in xrange(4):
        times = []
        for size in size1:
            platforms = cl.get_platforms()
            print 'The platforms detected are:'
            print '---------------------------'
            for platform in platforms:
                print platform.name, platform.vendor, 'version:', platform.version

            # List devices in each platform
            for platform in platforms:
                print 'The devices detected on platform', platform.name, 'are:'
                print '---------------------------'
                for device in platform.get_devices():
                    print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
                    print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
                    print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
                    print 'Maximum work group size', device.max_work_group_size
                    print '---------------------------'

            # Create a context with all the devices
            devices = platforms[0].get_devices()
            context = cl.Context(devices)
            print 'This context is associated with ', len(context.devices), 'devices'

            # Create a queue for transferring data and launching computations.
            # Turn on profiling to allow us to check event times.
            queue = cl.CommandQueue(context, context.devices[0],
                                    properties=cl.command_queue_properties.PROFILING_ENABLE)
            print 'The queue is using the device:', queue.device.name

            curdir = os.path.dirname(os.path.realpath(__file__))

            program = cl.Program(context, open('aniso_openCL.cl').read()).build(options=['-I', curdir])

            # Load the image to numpy array.
            if npzfile:
                host_image = np.load(image_path)['image'].astype(np.float32).copy()
            else:
                host_image = imread(image_path).astype(np.float32).copy()

            # Create a numpy array of the image size.
            host_image_filtered = np.zeros_like(host_image)

            gpu_image_a = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
            gpu_image_b = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)


            local_size = (size, 2)  

            global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)]) 
            if method not in {0,3}: global_size = (global_size[0], local_size[1])

            width = np.int32(host_image.shape[1])
            height = np.int32(host_image.shape[0])
            halo = np.int32(1)

            # Set up a (N+2 x N+2) local memory buffer.
            # +2 for 1-pixel halo on all sides, 4 bytes for float.
            local_memory = cl.LocalMemory(4 * (local_size[0] + 2) * (local_size[1] + 2))
            #local_memory = cl.LocalMemory(4 * (local_size[1] + 2*halo) * (local_size[0] + 2*halo))
            # Each work group will have its own private buffer.
            buf_width = np.int32(local_size[0] + 2)
            buf_height = np.int32(local_size[1] + 2*halo)
            

            # Send image to the device, non-blocking
            cl.enqueue_copy(queue, gpu_image_a, host_image, is_blocking=False)

            l = 0.2
     
            with Timer() as t1:
                for iter in range(num_iters):
                    # Parallel by Block with the global size covering the whole image.
                    if method == 0:
                        program.aniso_blockparallel(queue, global_size, local_size,
                                           gpu_image_a, gpu_image_b, local_memory,
                                           width, height,
                                           buf_width, buf_height, halo, np.float32(l));

                    # Parallel by column. Each column should have one buffer only.
                    elif method == 1:
                        program.aniso_colparallel(queue, global_size, local_size,
                                           gpu_image_a, gpu_image_b, local_memory,
                                           width, height,
                                           buf_width, buf_height, halo, np.float32(l));

                    # Index trick to avoid rereading values to the buffer.
                    elif method == 2:
                        program.aniso_reusedparallel(queue, global_size, local_size,
                                           gpu_image_a, gpu_image_b, local_memory,
                                           width, height,
                                           buf_width, buf_height, halo, np.float32(l));

                    # Read from the global memory directly.
                    else:
                        program.aniso_nobufferparallel(queue, global_size, local_size,
                                           gpu_image_a, gpu_image_b, 
                                           width, height,
                                           buf_width, buf_height, halo, np.float32(l));

                    # swap filtering direction
                    gpu_image_a, gpu_image_b = gpu_image_b, gpu_image_a

                
            print("{} seconds for 40 filter passes using vectorization in openCL.".format(t1.interval))
            times.append(t1.interval)

            cl.enqueue_copy(queue, host_image_filtered, gpu_image_a, is_blocking=True)
        pylab.plot(size1, times, label = labels[method], marker = ".")
        pylab.title("Effciency vs Different Parallelization")
        pylab.xlabel("Local Size Width (Height = 2)")
        pylab.ylabel("Time")
        pylab.legend(loc="best")
        
    pylab.show()
    return host_image_filtered

'''
@param image_path : path of the image
@param npzfile : a boolean indicating whether the image is an npz file.   
@param num_iters : number of iterations
'''
def run_serial_program(image_path, npzfile, num_iters):
    if npzfile:
        host_image = np.load(image_path)['image'].astype(np.float32).copy()
    else:
        host_image = imread(image_path).astype(np.float32).copy()
    out_image = None
    with Timer() as t2:
        out_image = aniso.anisodiff_vec(host_image, num_iters)

    print("{} seconds for 10 filter passes using vectorization in numpy.".format(t2.interval))
    return out_image

def plot_image(filtered_image):
    pylab.figure()
    pylab.gray()
    pylab.imshow(filtered_image)
    pylab.title('after - zoom')
    pylab.show()

if __name__ == '__main__':
    res = run_parallel_program('image.npz', True, 40)
    plot_image(res[1200:1800, 3000:3500])
