import gc
import time

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)



class Helper_Multiprocessing:
    
    def __init__(self, all_processes: list[multiprocessing.Process], shots_at_a_time: int=1, wait_before_starting_the_next_process: int=0):
        
        self.all_processes = all_processes
        self.shots_at_a_time = shots_at_a_time
        
        self.wait_before_starting_the_next_process = wait_before_starting_the_next_process
        
        self.completed_processes = 0
        self.running_processes = 0
        self.current_index = 0
        
        return
    
    
    def check_running_processes(self):
        
        self.running_processes = 0
        for process in self.all_processes:
            if process.is_alive():
                self.running_processes += 1
        
        gc.collect()
        
        return
    
    
    def run_next_process(self):
        
        if (self.wait_before_starting_the_next_process>0) & (self.current_index>0):
            time.sleep(self.wait_before_starting_the_next_process)
        
        self.all_processes[self.current_index].start()
        self.current_index += 1
        
        return
    
    
    def run_all_processes(self, only_exit_when_all_processes_are_finished: bool=False):
        
        i = 0
        while self.current_index < len(self.all_processes):
            self.check_running_processes()
            if self.running_processes < self.shots_at_a_time:
                self.run_next_process()
            
            i = (i+1) % 100
            if i == 0:
                print('\n\nRunning process {}, {}/{}.\n\n'.format(self.running_processes, self.current_index, len(self.all_processes)))
                
        while self.running_processes > 0:
            self.check_running_processes()
            
        return
    
    