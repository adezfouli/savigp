# NB if started with job_hash which doesnt exist, will restart from scratch and not warn !!
def launch_qsub_job(params, 
    files_prefix = '/bigscratch/sb358/pygpstruct-chain/results/afac/',
    job_hash = None, 
    file_prefix_without_data_indices=True,
    repeat_runs=1):
    """
    params: dict of parameters to pass to prepare_from_data_chain.learn_predict_gpstruct_wrapper
    files_prefix : where all the files are put. Can end with / (directory) or not (common file prefix)
    job_hash : string or None. if None, creates new job folder. if string, reuse what's in that folder.
    file_prefix_without_data_indices : whether or not the file_prefix should be made to contain info on data (train, test) indices
    """
    import hashlib
    import time
    import subprocess
    import os
    import sys
    
    files_path = os.path.split(files_prefix)[0]
    try:
        print('making path %s' % files_path)
        os.makedirs(files_path)
    except OSError as err:
        if err.errno!=17:
            raise

    if (job_hash == None):
        hash_string = hashlib.sha1()
        hash_string.update(str(time.time()))
        job_hash = hash_string.hexdigest()[:10]
    
    log_files_prefix = files_prefix + job_hash
    
    # turn params to a filename
    
    params_for_filename = params.copy()
    if file_prefix_without_data_indices:
        params_for_filename.pop('data_indices_test')
        params_for_filename.pop('data_indices_train')
    params_for_filename.pop('data_folder')
    param_as_filename = '++'.join(['%s=%s' % (k,params_for_filename[k]) for k in sorted(params_for_filename.keys())])
    param_as_filename = param_as_filename.replace(' ', '').replace('(','').replace(')','').replace(',','-').replace("'", '').replace('{', '').replace('}','')
    python_job_log_file_prefix = log_files_prefix + "." + param_as_filename

    params['result_prefix'] = "'" + python_job_log_file_prefix + ".'" # will be passed inside string command line
    params['console_log'] = 'False'

    python_script_name = "%s.py" % log_files_prefix
    with open(python_script_name, "w") as f:
        f.write('import sys; sys.path.append(\'/home/mlg/sb358/pygpstruct/src\'); ' + 
                'import numpy as np; import prepare_from_data_chain; ' + 
                'import util; util.stop_check((12-0.1)*3600); ' + # stop 360 sec before SGE job will be killed (which is after 12h) to reduce chances of being killed in the middle of a state save operation
                'prepare_from_data_chain.learn_predict_gpstruct_wrapper(%s, stop_check=util.stop_check);\n'
                % (', '.join(['%s=%s' % (k, params[k]) for k in sorted(params.keys())]))) 
    
    shell_script_name = "%s.sh" % log_files_prefix
    with open(shell_script_name, "w") as f:
        f.write("#!sh\n")
        # f.write("source activate py3")
        f.write('python %s >%s.out 2>%s.err\n' # should replace > by >> 
                % (python_script_name, python_job_log_file_prefix, python_job_log_file_prefix))            

    for rerun in range(repeat_runs):
        ssh_command_as_list = ["ssh", "fear", "qsub", "-o", "%s.qsub.out" % (log_files_prefix), "-e", "%s.qsub.err" % (log_files_prefix)]
        if (rerun > 0):
            ssh_command_as_list.extend(['-hold_jid', str(job_id)]) # if second or higher in a sequence of runs, wait for the previous job to finish
        ssh_command_as_list.append(shell_script_name)
        
        process = subprocess.Popen(ssh_command_as_list, 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       cwd= files_path,
                                       )
                # qsub -o $model.fold_$fold.out -e $model.fold_$fold.err $model.sh $fold /bigscratch/sb358/image_segmentation/model_$model.train50/fold_$fold/
        
        # Poll process for new output until finished # from # http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
        while True:
            nextline = process.stdout.readline().decode('utf-8')
            import re
            match = re.match('Your job (\d+) \(".+"\) has been submitted', nextline)
            if (match != None):
                job_id = match.group(1)
                with open("%s.rerun_%s.qsub.jobid" % (log_files_prefix, str(rerun)), "w") as f:
                    f.write(job_id)
            if nextline == '' and process.poll() != None:
                break
            sys.stdout.write(nextline)
            sys.stdout.flush()

        output = process.communicate()[0] # communicate returns (stdout, stderr)
        exitCode = process.returncode

        if (exitCode == 0):
            pass # return output
        else:
            print(output)
            raise subprocess.CalledProcessError(exitCode, cmd='TODO', output=output)