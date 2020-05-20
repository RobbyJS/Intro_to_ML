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


if __name__=='__main__':

    top_c = ["Orchidea","Cerezo japones","Rosa blanca de los desiertos de Almeria"]
    top_p = [0.36,0.3,0.1]

    print("\n")
    print("Below are the predictions obtained from the model:")
    pprint(top_c,top_p)
    print("\n")