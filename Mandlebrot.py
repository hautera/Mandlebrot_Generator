import tensorflow as tf
import numpy as np
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
import argparse


#
#Saves the set a as a picture as file name
#a is the coordinates to be drawn
#file_name is the file the coordinates will be saved to
#
def display_fractal( a, file_name= 'mandlebrot.jpg' ):
    a_cyclic = ( 6.28 * a / 20.0 ).reshape( list( a.shape ) + [1] )
    img = np.concatenate( [ 10 + 20 * np.cos( a_cyclic ),
                                             30 + 50 * np.sin( a_cyclic ),
                                             55 - 80 * np.cos( a_cyclic ) ], 2 )
    print( a.max() )
    img[ a == a.max() ] = 0
    a = img
    a = np.uint8( np.clip( a, 0, 255 ))
    f = BytesIO()
    PIL.Image.fromarray( a ).save( file_name )


#
#Finds Values of the madlebrot set and saves them as a jpeg picture
#
def main():

    #set up the argument parser
    desc = "Finds Values of the madlebrot set and saves them as a jpeg picture"
    parser = argparse.ArgumentParser( description = desc )
    parser.add_argument( '--file', dest='file_name', required=False,
                                        help="The name of the file this program will save to. By default this program saves to mandlebrot.jpg" )

    parser.add_argument( '--res', dest='res', required=False,
                                        help="The scale of the resolution of the image to be made, default is 2" )
    parser.add_argument('-v', action='store_true')

    args = parser.parse_args()
    verbose = args.v

    if verbose:
        print("Setting up session")
    sess = tf.InteractiveSession()
    if args.res:
        if verbose:
            print("Caculations will be done to {} decimal presision".format(1 / ( 10 ** int( args.res ))))
        Y, X = np.mgrid[-1.3:1.3:1 / ( 10 ** int(args.res) ), -2:1: 1/(10 ** int(args.res))]
    else:
        if verbose:
            print("Caculations will be done to 0.005 decimal presision")
        Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
    Z = X + 1j*Y

    if verbose:
        print("Setting up complex plane")
    xs = tf.constant( Z.astype( np.complex64 ))
    zs = tf.Variable( xs )
    ns = tf.Variable( tf.zeros_like( xs, tf.float32 ))
    tf.global_variables_initializer().run()

    zs_ = zs * zs + xs
    not_diverged = tf.abs( zs_ ) < 4

    if verbose:
        print("Assigning step function")
    step = tf.group(
        zs.assign( zs_ ),
        ns.assign_add( tf.cast( not_diverged, tf.float32 ))
        )

    for i in range( 200 ):
        if verbose:
            print("Running step {}".format( i ))
        step.run()

    if args.file_name:
        display_fractal( ns.eval(), file_name = args.file_name )
    else:
        display_fractal( ns.eval() )

if __name__ == '__main__':
    main()
