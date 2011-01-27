Executable = q4/build_dt.pl
Universe = vanilla
getenv = true
output = acc_file.50_0
error = 50_0.err
Log = 50_0.log
arguments = "examples/train.vectors.txt examples/test.vectors.txt 50 0 model_file.50_0 sys_file.50_0"
transfer_executable = false
Queue
