#!/bin/sh
#
# compresses files associated with a COMP 150-02 homework using zip (or tar if 
# zip is not available)
#
# note: tar compression results in error 'file changed as we read it' for 
# current directory. this is OK

find . -name '*.ipynb' | xargs ipython nbconvert --to html
hw=hw3
exclude=("*.git*" "*datasets/cifar-10-batches-py*"                      \
    "*.ipynb_checkpoint*" "*README.md" "*collectSubmission.sh"         \
    "*requirements.txt")

# check to see if system has zip or tar
if command -v zip >/dev/null 2>&1
then                                    # zip

    rm -f "$hw".zip

    exclude_option="-x"
    for f in "${exclude[@]}"
    do
        exclude_option="$exclude_option"" ""\"""$f""\""
    done

    cmd="zip -r ""$hw"".zip . ""$exclude_option"

else                                    # tar

    rm -f "$hw".tar.gz

    exclude_option=""
    for f in "${exclude[@]}"
    do
        exclude_option="$exclude_option""--exclude='"$f"' "
    done

    cmd="tar ""$exclude_option"" -zcf ""$hw"".tar.gz ."

fi

eval $cmd
