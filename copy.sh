arrayName=()
while IFS='' read -r line || [[ -n "$line" ]]; do
    # echo "Text read from file: $line"
    arrayName+="$line"
done < "$1"

# arrayName=( "chaofeng.txt" "test2.txt" "test.txt" )
for i in "${arrayName[@]}"
do
   scp $i chaofeng@Linux.cs.uchicago.edu:/home/chaofeng/Documents/practicum/copy_images/images
   # echo $i
done
