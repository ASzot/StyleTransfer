# Simple Style Transfer

When I was looking for examples of style transfer in TensorFlow I found a lot
of repos that were overcomplicated with many files and working parts. Rather
than having extensible and "use" ready code I wanted code that was just concise
and easy to read so I could understand style transfer. 

So rather than have a million configurable command line arguments and a bunch
of utility files there's just one `main.py` taking the simpliest approach to
style transfer with comments and explanations all in under 150 lines.
`vgg19.py` simply defines a pretrained VGG network. You need to download the
VGG weights at http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat.

See the accompanying blog post for this at https://www.andrewszot.com/blog/machine_learning/deep_learning/style_transfer.
