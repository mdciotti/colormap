for file in 01/*
    do
        if [ -f $file ]
        then
            #tail +19c $file > $file.truncated && mv $file.truncated $file
            tail +20c $file > $file.truncated1
            head -c $(($(stat -f '%z' $file.truncated1)-1)) $file.truncated1 > $file.truncated2
            rm $file.truncated1
            mv $file.truncated2 $file
        fi
    done

