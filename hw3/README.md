# Homework 
Convolutional Networks

Details about this assignment can be found [on the course webpage](https://comp150dl.github.io/hw/).

Find all TODO items by running this command in the top level dir:

> find . \( -name '*.ipynb'  -o -name '*.py' \)  | xargs grep -n TODO | grep -v image | grep -v checkpoint

Note: 

1) You'll need a Cython extension for part of this assignment. Check the 
requirements file and update your environment accordingly. (You may have already
done this for homework 2.)

2) You'll need the layers you implemented for homework 2 in this assignment. 
Copy the layers.py file from the hw2/ directory of the previous assignment into 
the hw3/ folder of this assignment.

## Submission

To prepare your work for submission, double check that all notebooks
cells have been run to your satisfaction!

If you are on Unix-type system, you can run the command:

> ./collectSubmission.sh

This will create a hw3.zip (or hw3.tar.gz) file. 

If you are not on a Unix-type system, please create a zip or tar.gz archive of 
your homework. You do not need to include the data files.

Submit your compressed archive by emailing to comp150dl@gmail.com.
