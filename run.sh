source compile.sh
dir=output/%1
echo $dir
mkdir $dir
output/Runner $dir 2>&1 | tee $dir/err
