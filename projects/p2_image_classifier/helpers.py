import os
from itertools import cycle, islice

def pprint(list_c,list_p):
    """Function to print the flower classes together with their probabilities, in a table for the 
    command output"""
    # Get longest name in the list
    max_l = 0
    for c in list_c:
        max_l = max(max_l,len(c))
    
    max_l = max_l+2
    cell_fmt_class_title = "{:>"+str(max_l)+"}"
    cell_fmt_prob_title = "{:<15}"
    row_template = cell_fmt_class_title + " | " +cell_fmt_prob_title
    

    print(row_template.format("Classes","Probabilities"))
    print("*"*(15+max_l+2))
    row_template = cell_fmt_class_title + " | " + "{:<15.3f}"
    for c, p in zip(list_c,list_p):
        print(row_template.format(c,p))
    print("*"*(15+max_l+2))

def pprint_train(epochs,total_e,steps,batch_size,print_every,t_loss,v_loss,v_accuracy):
        """Function to print the training progress"""
        title_template = "|{:^7}|{:^9}|{:^8}|{:^8}|{:^12}|"
        title_line = title_template.format("Epochs","Batches","Training","Validat.","Validat.")
        
        batch_size = (batch_size//print_every)*print_every
        total_steps = steps + (epochs-1)*batch_size
        epochs = [epochs + i//batch_size for i in range(1,total_steps+1,print_every)]
        # steps = list(range(print_every,steps+1,print_every))
        steps = list(islice(cycle(range(print_every,batch_size+1,print_every)), None, len(epochs)))
        # steps = [i%(batch_size) for i in range(print_every,total_steps+1,print_every)]

        # print(f"Vector lengths: epochs - {len(epochs)}, steps -{len(steps)},t loss - {len(t_loss)}, v loss - {len(v_loss)}, acc - {len(v_accuracy)}")
        print("-"*len(title_line))
        print(title_line)
        print(title_template.format("","","Losses","Losses","Accuracy"))        
        print("-"*len(title_line))
        row_template = "|{:>3d}{:^s}{:<3d}|{:^9d}|{:^8.3f}|{:^8.3f}|{:^12.3f}|"
        
        for ii in range(len(t_loss)):
            print(row_template.format(epochs[ii],"/",total_e,steps[ii],t_loss[ii],v_loss[ii],v_accuracy[ii]))
        
        print("*"*len(title_line))

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


if __name__=='__main__':
    #-----------------------------------------------------------------------------------#
    # Test of pprint function - Uncomment if needed
    # top_c = ["Orchidea","Cerezo japones","Rosa blanca de los desiertos de Almeria"]
    # top_p = [0.36,0.3,0.1]

    # print("\n")
    # print("Below are the predictions obtained from the model:")
    # pprint(top_c,top_p)
    # print("\n")

    #-----------------------------------------------------------------------------------#
    # Test of progress_bar function
    # import time
    # import os
    # # A List of Items    
    # items = list(range(0, 57))
    # l = len(items)

    # # Initial call to print 0% progress
    # _ = os.system('cls')
    # _ = os.system('clear')
    # printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    # for i, item in enumerate(items):
    #     # Do stuff...
    #     time.sleep(0.1)
    #     # Update Progress Bar
    #     printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    # pass

    #-----------------------------------------------------------------------------------#
    # Test of pprint function - Uncomment if needed
    import numpy as np
    _ = os.system('cls')
    _ = os.system('clear')
    epochs = 4
    total_ep = 15
    steps = 80
    batch_size = 110
    print_every = 20

    batch_size = (batch_size//print_every)*print_every
    total_steps = steps + (epochs-1)*batch_size

    len_v = len(range(1,total_steps+1,print_every))
    t_loss , v_loss, v_accuracy = np.random.rand(len_v), np.random.rand(len_v), np.random.rand(len_v)
    pprint_train(epochs,total_ep,steps,batch_size,print_every,t_loss,v_loss,v_accuracy)