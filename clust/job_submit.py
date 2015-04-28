# -*- coding: utf-8 -*-
import random
import string
import sys
from popen2 import popen2
import time

#many = int(sys.argv[1])
from subprocess import Popen, PIPE, STDOUT

many=1
for i in range(0, many):

    # Open a pipe to the qsub command.

    name = 'bost_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(3))
    # Customize your options here
    job_name = "adez_%s" %(name)
    walltime = "4:00:00"
    processors = "nodes=1:ppn=64"
    command = "./run_job.sh %s" % (str(i))
    job_string = """#!/bin/bash
#PBS -N %s
#PBS -l walltime=%s
#PBS -l %s
#PBS -q short64
#PBS -o /home/z3510738/code/output/%s.out
#PBS -e /home/z3510738/code/error/%s.err
#PBS -M a.dezfouli@unsw.edu.au
#PBS -m unsw_job_finished
cd $PBS_O_WORKDIR
cd /home/z3510738/code/savigp/clust/
chmod +x ./run_job.sh
%s""" % (job_name, walltime, processors, job_name, job_name, command)
#    check_dir_exists('../qsub/')
    out = open('../qsub/auto_run.pbs', 'w')
    out.write(job_string)
    out.close()
    time.sleep(2.0)
    p = Popen(['qsub', '../qsub/auto_run.pbs'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    output = p.communicate()[0]
    print output
    print job_string


