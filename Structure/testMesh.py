import get1DMesh as get1DMesh
import getPlot as getPlot
import getMK as getMK
import getBC as getBC
import numpy as np

#Initial structure
test_nodecoord = np.array([[0,0],[0,1],[0,2],[1,2],[1,1],[1,0]])
test_B = np.array ([[1,2],[2,3],[3,4],[4,5],[2,5],[5,6]])

#Fixed BC
bc = {0:1,5:1} 

getPlot.getPlot(test_nodecoord,test_B,True,True,10)

# 1D MESH
h = 1 #mesh size
new_nodecoord,new_B,itn = get1DMesh.get1DMesh(test_nodecoord, test_B, h)
getPlot.getPlot(new_nodecoord,new_B,True,True,10)     
        
# Global Matrix
ele_A = np.ones(new_B.shape[0])
ele_m = np.ones(new_B.shape[0])
ele_I = np.ones(new_B.shape[0])
ele_E = np.ones(new_B.shape[0])

ele_A[0] = 2
ele_E[0] = 2
ele_A[new_B.shape[0] - 1] = 2
ele_E[new_B.shape[0] - 1] = 2
K,M = getMK.getMK_FEM(new_nodecoord,new_B, ele_m, ele_E, ele_A, ele_I)
#print("stiffness matrix: ", K)
#print("mass matrix: ", M)     
#print("old to new ", itn)  


#Transformed Matrix
M_bc,K_bc,inact_num,glb_to_bc = getBC.MK_Reduction(M, K, bc, itn)

	

 






