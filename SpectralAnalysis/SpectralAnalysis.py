import pyTempNet as tn
import igraph 
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

def MatrixToLatex(A, names):       
    latex = "\[ \n\mathbf{\mathcal{L}} = \n\kbordermatrix{\n"        
    for i in range(len(A)):
        latex = latex + '&' + names[i]
    latex = latex + '\\\\' + '\n'
    for i in range(len(A)):
        latex = latex + str(names[i]) + '&'
        for j in range(len(A)):
            if float(A[i,j]).is_integer():
                latex = latex + str(int(A[i,j]))
            else:
                latex = latex + str(A[i,j])
            if j!=len(A)-1:
                latex = latex + "&"
        if i!=len(A)-1:
            latex = latex + '\\\\'+ '\n'
    latex = latex + "\n}\n\]"
    return latex

def plotSpectrum(t, filename='spectrum'):
    L = tn.Laplacian(t)
    w, v = spl.eig(L, left=True, right=False)
    evals_sorted = np.sort(w)

    L_null = tn.Laplacian(t, model='NULL')
    w_null, v_null = spl.eig(L_null, left=True, right=False)
    evals_sorted_null = np.sort(w_null)

    plt.clf()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)       
    plt.xlabel('$i$', fontsize=30)
    plt.ylabel(r'$(w_i)_i$', fontsize=30)
    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(left=0.25)
    plt.scatter(range(len(evals_sorted)), np.real(evals_sorted), color="r")
    plt.scatter(range(len(evals_sorted_null)), np.real(evals_sorted_null), color="b")
    plt.savefig(filename)
    plt.close()

def plotFiedler(t, filename='fiedler'):
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
    plt.savefig(filename)
    plt.close()


##############################
# London tube data
##############################
t_lt = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/tube_flows_scc.tedges', sep= ' ')
t_lt.extractTwoPaths()

# Plot first-order network
g1 = t_lt.igraphFirstOrder()

visual_style = {}
visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["edge_curved"] = .5
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_size"] = 10
visual_style["layout"] = g1.layout_auto()

igraph.plot(g1, 'LT_g1.png', **visual_style)
igraph.plot(g1, 'LT_g1.pdf', **visual_style)

# Export video of diffusion dynamics
tn.exportDiffusionMovieFramesFirstOrder(t_lt, 'frames/LT_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t_lt, 'frames/LT_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

# Export video of empirical and shuffled temporal evolution
visual_style["vertex_color"] = "lightblue"
t_lt.exportMovieFrames('frames/LT', visual_style, realtime=False, maxSteps=400)

shuffled = t_lt.ShuffleEdges()
shuffled.exportMovieFrames('frames/LT_shuffled', visual_style, realtime=False, maxSteps=400)

# Plot second-order networks
g2n = t_lt.igraphSecondOrderNull()
visual_style = {}
visual_style["edge_width"] = [x**.1 for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 8
visual_style["layout"] = g2n.layout_auto()
igraph.plot(g2n, 'LT_g2n.pdf', **visual_style)
igraph.plot(g2n, 'LT_g2n.png', **visual_style)

g2 = t_lt.igraphSecondOrder()
visual_style["edge_width"] = [x**.1 for x in g2.es()["weight"]]
igraph.plot(g2, 'LT_g2.pdf', **visual_style)
igraph.plot(g2, 'LT_g2.png', **visual_style)

# Spectral analysis
print('lambda_2(L) =', tn.AlgebraicConn(t_lt))
print('lambda_2(L_null) =', tn.AlgebraicConn(t_lt, model='NULL'))

plotFiedler(t_lt, 'LT_fiedler.pdf')
plotSpectrum(t_lt, 'LT_spectrum.pdf')

# Verify that leading eigenvector of both is the same
L = tn.Laplacian(t_lt)
w, v = spl.eig(L, left=True, right=False)
leading = v[:,np.argsort(np.absolute(w))][:,0]

L_null = tn.Laplacian(t_lt, model='NULL')
w_null, v_null = spl.eig(L_null, left=True, right=False)
leading_null = v_null[:,np.argsort(np.absolute(w_null))][:,0]

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

##############################
# Ants data set
##############################
t_an = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/ants-1-1_agg_6s_scc.tedges', sep= ' ')
t_an.extractTwoPaths()

# Plot first-order network
g1 = t_an.igraphFirstOrder()

visual_style = {}
visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["edge_curved"] = .2
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_size"] = 10
visual_style["layout"] = g1.layout_auto()

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
visual_style["edge_curved"] = .2
visual_style["vertex_size"] = 8
visual_style["layout"] = g2n.layout_auto()
igraph.plot(g2n, 'AN_g2n.pdf', **visual_style)
igraph.plot(g2n, 'AN_g2n.png', **visual_style)

g2 = t_an.igraphSecondOrder()
visual_style["edge_width"] = [x**.1 for x in g2.es()["weight"]]
igraph.plot(g2, 'AN_g2.pdf', **visual_style)
igraph.plot(g2, 'AN_g2.png', **visual_style)

# Spectral analysis
print('lambda_2(L) =', tn.AlgebraicConn(t_an))
print('lambda_2(L_null) =', tn.AlgebraicConn(t_an, model='NULL'))

plotFiedler(t_lt, 'LT_fiedler.pdf')
plotSpectrum(t_an, 'AN_spectrum.pdf')

################################
#  Reality Mining
################################
t_rm = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/RealityMining_agg_300s_scc.tedges', sep= ' ')
t_rm.extractTwoPaths()

# Plot second-order networks
g2 = t_rm.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2.layout_auto()

visual_style["edge_width"] = [x**.5 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.05
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 5
igraph.plot(g2, 'RM_g2.pdf', **visual_style)
igraph.plot(g2, 'RM_g2.png', **visual_style)

# Plot first-order network
g1 = t_rm.igraphFirstOrder()

visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'RM_g1.pdf', **visual_style)
igraph.plot(g1, 'RM_g1.png', **visual_style)

# Export diffusion video
tn.exportDiffusionMovieFramesFirstOrder(t_rm, 'frames/RM_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t_rm, 'frames/RM_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

# Spectral analysis
print('lambda_2(L) =', tn.AlgebraicConn(t_rm))
print('lambda_2(L_null) =', tn.AlgebraicConn(t_rm, model='NULL'))

plotFiedler(t_rm, 'RM_fiedler.pdf')
plotSpectrum(t_rm, 'RM_spectrum.pdf')


################################
#  E-Mail
################################
t_em = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/manufacturing_30d_agg_3600_scc.tedges', sep= ' ')
t_em.extractTwoPaths()

# Plot first-order network
g1 = t_em.igraphFirstOrder()

visual_style = {}
visual_style["bbox"] = (900, 900)
visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_label"] = g1.vs["name"]
visual_style["edge_curved"] = .5
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_size"] = 25
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'EM_g1.png', **visual_style)
igraph.plot(g1, 'EM_g1.pdf', **visual_style)

# Export diffusion videos
tn.exportDiffusionMovieFramesFirstOrder(t_em, 'frames/EM_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t_em, 'frames/EM_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

# Plot temporal evolution
t.exportMovieFrames('frames/EM', visual_style, realtime=False)
t.exportMovieFrames('frames/EM_bursts', visual_style, realtime=True)

# Plot second-order networks
g2 = t_em.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2.layout_auto()
visual_style["edge_width"] = [x**.5 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10

igraph.plot(g2, 'EM_g2.pdf', **visual_style)
igraph.plot(g2, 'EM_g2.png', **visual_style)

# Spectral analysis
print('lambda_2(L) =', tn.AlgebraicConn(t_em))
print('lambda_2(L_null) =', tn.AlgebraicConn(t_em, model='NULL'))

plotFiedler(t_em, 'EM_fiedler.pdf')
plotSpectrum(t_em, 'EM_spectrum.pdf')

################################
#  Model data
################################
t_su = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/sigma0_75.trigram', sep= ' ', fformat='TRIGRAM')

# Plot second-order network
g2 = t_su.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2.layout_auto()
visual_style["edge_width"] = [x**.5 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
igraph.plot(g2, 'Model_075_g2.pdf', **visual_style)

# Plot first-order network
g1 = t.igraphFirstOrder()

visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'Model_g1.pdf', **visual_style)

# Export diffusion videos
tn.exportDiffusionMovieFramesFirstOrder(t_su, 'frames/Model_075_diffusion_t2', visual_style, steps = 200, initial_index=50, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t_su, 'frames/Model_075_diffusion_t1', visual_style, steps = 200, initial_index=50, model='NULL')

# Spectral analysis
print('lambda_2(L) =', tn.AlgebraicConn(t_su))
print('lambda_2(L_null) =', tn.AlgebraicConn(t_su, model='NULL'))

plotFiedler(t_su, 'Model_su_fiedler.pdf')
plotSpectrum(t_su, 'Model_su_spectrum.pdf')

# slow-down
t_sd = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/sigma-0_75.trigram', sep= ' ', fformat='TRIGRAM')

# Export diffusion video
tn.exportDiffusionMovieFramesFirstOrder(t_sd, 'frames/Model_-075_diffusion_t2', visual_style, steps = 200, initial_index=50, model='SECOND')

# Spectral analysis
print('lambda_2(L) =', tn.AlgebraicConn(t_sd))
print('lambda_2(L_null) =', tn.AlgebraicConn(t_sd, model='NULL'))

plotFiedler(t_sd, 'Model_sd_fiedler.pdf')
plotSpectrum(t_sd, 'Model_sd_spectrum.pdf')



################################
#  Example networks
################################
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

L_ex1 = tn.Laplacian(t_ex1, model = 'SECOND')
L_ex2 = tn.Laplacian(t_ex2, model = 'SECOND')

print(MatrixToLatex(L_ex1, g2.vs()["name"]))
print(MatrixToLatex(L_ex2, g2.vs()["name"]))

############################################
# Minimal example for non-Markovian temporal network
############################################

example = tn.TemporalNetwork()

example.addEdge('a', 'c', 1)
example.addEdge('c', 'd', 2)

example.addEdge('b', 'c', 3)
example.addEdge('c', 'e', 4)

example.addEdge('a', 'c', 5)
example.addEdge('c', 'd', 6)

example.addEdge('b', 'c', 7)
example.addEdge('c', 'e', 8)

A = np.matrix(list(t_ex1.igraphFirstOrder().get_adjacency()))
print('A =', A)
print('A^2 =', np.linalg.matrix_power(A, 2))
print('A^3 =', np.linalg.matrix_power(A, 3))

A_2 = np.matrix(list(t_ex1.igraphSecondOrder().get_adjacency()))
print('A_2 =', A_2)
print('A_2^2 =', MatrixToLatex(np.linalg.matrix_power(A_2, 2), names=t_ex1.igraphSecondOrder().vs()['name']))
