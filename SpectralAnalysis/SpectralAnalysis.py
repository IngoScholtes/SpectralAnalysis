import pyTempNet as tn
import igraph 
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

t = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/tube_flows_scc.tedges', sep= ' ')
t.extractTwoPaths()

g1 = t.igraphFirstOrder()

visual_style = {}
visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["edge_curved"] = .5
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_size"] = 10
visual_style["layout"] = g1.layout_auto()

# Export plots of first-order aggregate network
igraph.plot(g1, 'LT_g1.png', **visual_style)
igraph.plot(g1, 'LT_g1.pdf', **visual_style)

# Export video of diffusion dynamics
tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/LT_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/LT_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

# Export video of temporal evolution
visual_style["vertex_color"] = "lightblue"
t.exportMovieFrames('frames/LT', visual_style, realtime=False, maxSteps=400)

shuffled = t.ShuffleEdges()
shuffled.exportMovieFrames('frames/LT_shuffled', visual_style, realtime=False, maxSteps=400)

# Plot second-order networks
g2n = t.igraphSecondOrderNull()
visual_style = {}
visual_style["edge_width"] = [x**.1 for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 8
visual_style["layout"] = g2n.layout_auto()
igraph.plot(g2n, 'LT_g2n.pdf', **visual_style)
igraph.plot(g2n, 'LT_g2n.png', **visual_style)

g2 = t.igraphSecondOrder()
visual_style["edge_width"] = [x**.1 for x in g2.es()["weight"]]
igraph.plot(g2, 'LT_g2.pdf', **visual_style)
igraph.plot(g2, 'LT_g2.png', **visual_style)

# Detailed spectral analysis
fiedler = tn.Measures.FiedlerVector(t)
fiedler_null = tn.Measures.FiedlerVector(t, model= 'NULL')

plt.clf()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)       
plt.xlabel('$i$', fontsize=30)
plt.ylabel(r'$(v_2)_i$', fontsize=30)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
plt.scatter(range(len(fiedler)), np.real(fiedler), color="r")
plt.scatter(range(len(fiedler_null)), np.real(fiedler_null), color="b")
plt.savefig("LT_fiedler.pdf")
plt.close()

L = tn.Laplacian(t)
w, v = spl.eig(L, left=True, right=False)
leading = v[:,np.argsort(np.absolute(w))][:,0]

L_null = tn.Laplacian(t, model='NULL')
w, v = spl.eig(L_null, left=True, right=False)
leading_null = v[:,np.argsort(np.absolute(w))][:,0]

plt.clf()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)       
plt.xlabel('$i$', fontsize=30)
plt.ylabel(r'$(v_1)_i$', fontsize=30)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
plt.scatter(range(len(leading)), np.real(leading), color="r")
plt.scatter(range(len(leading_null)), np.real(leading_null), color="b")
plt.savefig("LT_leading.pdf")
plt.close()

print(tn.AlgebraicConn(t))
print(tn.AlgebraicConn(t, model='NULL'))


# Do the same for the ants
t_an = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/ants-1-1_agg_6s_scc.tedges', sep= ' ')
t_an.extractTwoPaths()

g1 = t_an.igraphFirstOrder()

visual_style = {}
visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["edge_curved"] = .5
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_size"] = 10
visual_style["layout"] = g1.layout_auto()

# Export plots of first-order aggregate network
igraph.plot(g1, 'AN_g1.png', **visual_style)
igraph.plot(g1, 'AN_g1.pdf', **visual_style)

# Export video of diffusion dynamics
tn.exportDiffusionMovieFramesFirstOrder(t_an, 'frames/AN_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t_an, 'frames/AN_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

# Export video of temporal evolution
visual_style["vertex_color"] = "lightblue"
t_an.exportMovieFrames('frames/AN', visual_style, realtime=False, maxSteps=400)
t_an.exportMovieFrames('frames/AN_bursts', visual_style, realtime=True, maxSteps=400)

shuffled = t_an.ShuffleEdges()
shuffled.exportMovieFrames('frames/AN_shuffled', visual_style, realtime=False, maxSteps=400)

# Plot second-order networks
g2n = t_an.igraphSecondOrderNull()
visual_style = {}
visual_style["edge_width"] = [x**.1 for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 8
visual_style["layout"] = g2n.layout_auto()
igraph.plot(g2n, 'AN_g2n.pdf', **visual_style)
igraph.plot(g2n, 'AN_g2n.png', **visual_style)

g2 = t_an.igraphSecondOrder()
visual_style["edge_width"] = [x**.1 for x in g2.es()["weight"]]
igraph.plot(g2, 'AN_g2.pdf', **visual_style)
igraph.plot(g2, 'AN_g2.png', **visual_style)

print(tn.AlgebraicConn(t_an))
print(tn.AlgebraicConn(t_an, model='NULL'))

################################
#  Reality Mining
################################
t = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/RealityMining_agg_300s_scc.tedges', sep= ' ')

t.extractTwoPaths()

g2 = t.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2.layout_auto()

visual_style["edge_width"] = [x**.5 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 5
igraph.plot(g2, 'RM_g2.pdf', **visual_style)

g1 = t.igraphFirstOrder()

visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'RM_g1.pdf', **visual_style)

tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/RM_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/RM_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')


# Do the same for the E-mail network
t = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/manufacturing_30d_agg_3600_scc.tedges', sep= ' ')

t.extractTwoPaths()

g1 = t.igraphFirstOrder()

visual_style = {}
visual_style["bbox"] = (900, 900)
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_label"] = g1.vs["name"]
visual_style["edge_curved"] = .5
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_size"] = 25
visual_style["layout"] = g1.layout_auto()

igraph.plot(g1, 'EM_g1.png', **visual_style)

t.exportMovieFrames('frames/EM', visual_style, realtime=False)
t.exportMovieFrames('frames/EM_bursts', visual_style, realtime=True)

g2 = t.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2.layout_auto()

visual_style["edge_width"] = [x**.5 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
igraph.plot(g2, 'EM_g2.pdf', **visual_style)

g1 = t.igraphFirstOrder()

visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'EM_g1.pdf', **visual_style)

tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/EM_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/EM_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

fiedler = tn.Measures.FiedlerVector(t)
fiedler_null = tn.Measures.FiedlerVector(t, model= 'NULL')

plt.clf()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)       
plt.xlabel('$i$', fontsize=30)
plt.ylabel(r'$(v_2)_i$', fontsize=30)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
plt.scatter(range(len(fiedler)), np.real(fiedler), color="r")
plt.scatter(range(len(fiedler_null)), np.real(fiedler_null), color="b")
plt.savefig("RM_fiedler.pdf")
plt.close()

print(tn.AlgebraicConn(t))
print(tn.AlgebraicConn(t, model='NULL'))


g2n = t.igraphSecondOrderNull()
visual_style["edge_width"] = [x**.2 for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 5
igraph.plot(g2n, 'RM_g2n.png', **visual_style)

tn.exportDiffusionMovieFrames(g2n, 'frames/RM_g2n_diffusion', visual_style, steps = 1000, initial_index=0)





# And finally for the model ... 
t = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/sigma0_75.trigram', sep= ' ', fformat='TRIGRAM')

g2 = t.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2.layout_auto()

visual_style["edge_width"] = [x**.5 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
igraph.plot(g2, 'Model_075_g2.pdf', **visual_style)

g1 = t.igraphFirstOrder()

visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'Model_g1.pdf', **visual_style)

tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/Model_075_diffusion_t2', visual_style, steps = 200, initial_index=50, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/Model_075_diffusion_t1', visual_style, steps = 200, initial_index=50, model='NULL')


t = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/sigma-0_75.trigram', sep= ' ', fformat='TRIGRAM')
tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/Model_-075_diffusion_t2', visual_style, steps = 200, initial_index=50, model='SECOND')


# Example networks
t_ex1 = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/example_1.trigram', sep= ' ', fformat='TRIGRAM')
t_ex2 = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/example_2.trigram', sep= ' ', fformat='TRIGRAM')

g2 = t_ex1.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2.layout_auto()
visual_style["bbox"] = (800,800)
visual_style["edge_width"] = [x**.5 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 2
visual_style["edge_curvature"] = .5
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_label"] = g2.vs()["name"]
visual_style["edge_label"] = g2.es()["weight"]
visual_style["vertex_size"] = 40
igraph.plot(g2, 'ex1_g2.png', **visual_style)

g2 = t_ex2.igraphSecondOrder()
igraph.plot(g2, 'ex2_g2.png', **visual_style)

def MatrixToLatex(A, names):   
    
    latex = "\[ \n\mathbf{\mathcal{L}} = \n\kbordermatrix{\n"
        
    for i in range(len(A)):
        latex = latex + '&' + names[i]
    latex = latex + '\\\\' + '\n'
    for i in range(len(A)):
        latex = latex + str(names[i]) + '&'
        for j in range(len(A)):
            if A[i,j].is_integer():
                latex = latex + str(int(A[i,j]))
            else:
                latex = latex + str(A[i,j])
            if j!=len(A)-1:
                latex = latex + "&"
        if i!=len(A)-1:
            latex = latex + '\\\\'+ '\n'
    latex = latex + "\n}\n\]"
    return latex    

L_ex1 = tn.Laplacian(t_ex1, model = 'SECOND')
L_ex2 = tn.Laplacian(t_ex2, model = 'SECOND')

print(MatrixToLatex(L_ex1, g2.vs()["name"]))
print(MatrixToLatex(L_ex2, g2.vs()["name"]))