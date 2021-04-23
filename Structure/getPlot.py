import matplotlib.pyplot as plt

# function for rendering plot for the structure
def getPlot(coord,B,show_ele_tag = None,show_node_tag = None,tag_size = 5,fig_size = (14,14),tag_marker = 'o'):
    fig, ax = plt.subplots(figsize = fig_size)
    
    for i in range(B.shape[0]):
        plt.plot((coord[B[i,0],0], coord[B[i,1],0]),(coord[B[i,0],1], coord[B[i,1],1]),marker = tag_marker,color = 'b')
        
        if show_ele_tag == True:
            ax.text((coord[B[i,0],0] + coord[B[i,1],0])/2,(coord[B[i,0],1] + coord[B[i,1],1])/2, str('element {}'.format(i+1)),fontsize = tag_size)
    for i in range(coord.shape[0]):
        if show_node_tag == True:
            ax.text(coord[i,0],coord[i,1]*0.99,str('  {}'.format(i+1)),fontsize = tag_size,color = 'r')
    
    

