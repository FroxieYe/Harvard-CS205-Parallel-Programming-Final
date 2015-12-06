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

def numpy_median(image, iterations=10):
    ''' filter using numpy '''
    for i in range(iterations):
        padded = np.pad(image, 1, mode='edge')
        stacked = np.dstack((padded[:-2,  :-2], padded[:-2,  1:-1], padded[:-2,  2:],
                             padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
                             padded[2:,   :-2], padded[2:,   1:-1], padded[2:,   2:]))
        image = np.median(stacked, axis=2)

    return image

if __name__ == '__main__':
    # List our platforms
    size1 = [8, 16, 32, 64, 128]
    labels = ["Block Parallel", "Column Parallel", "Column Reused Buffer Parallel", "No Buffer Parallel"]
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

            host_image = np.load('image.npz')['image'].astype(np.float32).copy()
            #host_image = imread("fc_lion.jpg").astype(np.float32).copy()
            host_image_filtered = np.zeros_like(host_image)

            gpu_image_a = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
            gpu_image_b = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)

            #256 pixels per work group, width > 48
            #0. blocks reread 1. reread 2. blocks small register buffer
            #register. instead of passing a pointer, we can pass in an array (10x34)

            local_size = (size, 2)  # 64 pixels per work group (32, 8)

            global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)]) #(global image size, 1)
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
            '''
            pylab.gray()

            pylab.imshow(host_image)
            pylab.title('original image')

            pylab.figure()
            pylab.imshow(host_image[1200:1800, 3000:3500])
            pylab.title('before - zoom')
            '''
           
            
            l = 0.2
            num_iters = 40
     
            with Timer() as t1:
                for iter in range(num_iters):
                    if method == 0:
                        program.aniso_blockparallel(queue, global_size, local_size,
                                           gpu_image_a, gpu_image_b, local_memory,
                                           width, height,
                                           buf_width, buf_height, halo, np.float32(l));
                    elif method == 1:
                        program.aniso_colparallel(queue, global_size, local_size,
                                           gpu_image_a, gpu_image_b, local_memory,
                                           width, height,
                                           buf_width, buf_height, halo, np.float32(l));
                    elif method == 2:
                        program.aniso_reusedparallel(queue, global_size, local_size,
                                           gpu_image_a, gpu_image_b, local_memory,
                                           width, height,
                                           buf_width, buf_height, halo, np.float32(l));
                    else:
                        program.aniso_nobufferparallel(queue, global_size, local_size,
                                           gpu_image_a, gpu_image_b, 
                                           width, height,
                                           buf_width, buf_height, halo, np.float32(l));

                    # swap filtering direction
                    gpu_image_a, gpu_image_b = gpu_image_b, gpu_image_a

                
            print("{} seconds for 40 filter passes using vectorization in openCL.".format(t1.interval))
            times.append(t1.interval)
            '''
            pylab.figure()
            pylab.imshow(host_image_filtered[1200:1800, 3000:3500])
            pylab.title('after - zoom')
            '''
            cl.enqueue_copy(queue, host_image_filtered, gpu_image_a, is_blocking=True)
        pylab.plot(size1, times, label = labels[method], marker = ".")
        pylab.title("Effciency vs Different Parallelization")
        pylab.xlabel("Local Size Width (Height = 2)")
        pylab.ylabel("Time")
        pylab.legend(loc="best")
        
    pylab.show()

    '''
    pylab.figure()
    pylab.imshow(host_image_filtered[1200:1800, 3000:3500])
    pylab.title('after - zoom lambda: ' + str(l) + ' #iterations: ' + str(num_iters))

    pylab.show()
    
    
    with Timer() as t2:
        out_image = aniso.anisodiff_vec(host_image)

    print("{} seconds for 10 filter passes using vectorization in numpy.".format(t2.interval))

    print host_image_filtered[:,-1]

    print out_image[:,-1]
    
    assert np.allclose(host_image_filtered, aniso.anisodiff_vec(host_image))
    '''
