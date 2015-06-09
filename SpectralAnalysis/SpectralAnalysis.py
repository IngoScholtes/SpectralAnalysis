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

def plotFiedler(t, filename='fiedler', model='SECOND', col='b'):
    fiedler = tn.Measures.FiedlerVector(t, model)
    #fiedler_null = tn.Measures.FiedlerVector(t, model= 'NULL')
    if model == 'NULL':
        fiedler = list(reversed(fiedler))
    plt.clf()
    plt.tick_params(axis='both', which='major', labelsize=20)
    #plt.xlabel('$i$', fontsize=20)
    #plt.ylabel(r'$(v_2)_i$', fontsize=20)
    plt.xlim(0, len(fiedler))    
    #plt.subplots_adjust(bottom=0.15)
    #plt.subplots_adjust(left=0.15)
    plt.scatter(range(len(fiedler)), np.real(fiedler), color=col, s=80)
    plt.tight_layout()
    #plt.scatter(range(len(fiedler_null)), np.real(fiedler_null), color="b")
    plt.savefig(filename)
    plt.close()

def plotVector(t, filename='vector', i=2):
    L = tn.Laplacian(t)
    w, v = spl.eig(L, left=True, right=False)

    L_null = tn.Laplacian(t, model='NULL')
    w_null, v_null = spl.eig(L_null, left=True, right=False)

    vector_null = v_null[:,np.argsort(np.absolute(w_null))][:,i]
    vector = v[:,np.argsort(np.absolute(w))][:,i]

    plt.clf()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)       
    plt.xlabel('$i$', fontsize=30)
    plt.ylabel(r'$(v_2)_i$', fontsize=30)
    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(left=0.25)
    plt.scatter(range(len(vector)), np.real(vector), color="r")
    plt.scatter(range(len(vector_null)), np.real(vector_null), color="b")
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
visual_style["edge_width"] = [np.log(x)/2 for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["edge_curved"] = .5
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_size"] = 15
visual_style["layout"] = g1.layout_kamada_kawai()

igraph.plot(g1, 'LT_g1.png', **visual_style)
igraph.plot(g1, 'LT_g1.pdf', **visual_style)

# Export video of diffusion dynamics
tn.exportDiffusionVideo(t_lt, 'LT_diffusion_t2.mp4', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionVideo(t_lt, 'LT_diffusion_t1.mp4', visual_style, steps = 200, initial_index=0, model='NULL')

# Export side-by-side comparison
tn.exportDiffusionComparisonVideo(t_lt, 'LT_diffusion_comparison.mp4', visual_style, steps = 200, initial_index=0)

# Export video of empirical and shuffled temporal evolution
visual_style["vertex_color"] = "lightblue"
t_lt.exportMovie('LT_temporal.mp4', visual_style, realtime=False, maxSteps=400)

shuffled = t_lt.ShuffleEdges()
shuffled.exportMovie('LT_shuffled_temporal.mp4', visual_style, realtime=False, maxSteps=400)

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

L = tn.Laplacian(t_lt)
np.linalg.matrix_rank(L)

L_n = tn.Laplacian(t_lt, model='NULL')
np.linalg.matrix_rank(L_n)

plotVector(t_lt, 'LT_v4.pdf', i=3)
plotVector(t_lt, 'LT_v5.pdf', i=4)
plotVector(t_lt, 'LT_v6.pdf', i=5)

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
visual_style["edge_width"] = [np.log(x) for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["edge_curved"] = .2
visual_style["edge_arrow_size"] = 0.5
visual_style["vertex_size"] = 25
visual_style["layout"] = g1.layout_auto()

igraph.plot(g1, 'AN_g1.png', **visual_style)
igraph.plot(g1, 'AN_g1.pdf', **visual_style)

# Export video of diffusion dynamics
tn.exportDiffusionMovieFramesFirstOrder(t_an, 'frames/AN_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t_an, 'frames/AN_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

# Export side-by-side comparison
tn.exportDiffusionComparisonVideo(t_an, 'AN_diffusion_comparison.mp4', visual_style, steps = 200, initial_index=0)

# Export video of temporal evolution
visual_style["vertex_color"] = "lightblue"
t_an.exportMovieFrames('frames/AN', visual_style, realtime=False, maxSteps=400)
t_an.exportMovieFrames('frames/AN_bursts', visual_style, realtime=True, maxSteps=400)


t_an.exportMovie('AN_temporal.mp4', visual_style, realtime=False, maxSteps=400)
shuffled = t_an.ShuffleEdges()
shuffled.exportMovie('AN_shuffled_temporal.mp4', visual_style, realtime=False, maxSteps=400)

# Plot second-order networks
g2n = t_an.igraphSecondOrderNull()
visual_style = {}
visual_style["edge_width"] = [x**.1 for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_color"] = "lightblue"
visual_style["edge_curved"] = .2
visual_style["vertex_size"] = 15
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
visual_style["vertex_size"] = 8
igraph.plot(g2, 'RM_g2.pdf', **visual_style)
igraph.plot(g2, 'RM_g2.png', **visual_style)

# Plot first-order network
g1 = t_rm.igraphFirstOrder()

visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_auto()
visual_style["vertex_size"] = 15
igraph.plot(g1, 'RM_g1.pdf', **visual_style)
igraph.plot(g1, 'RM_g1.png', **visual_style)

# Export diffusion video
tn.exportDiffusionMovieFramesFirstOrder(t_rm, 'frames/RM_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t_rm, 'frames/RM_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

tn.exportDiffusionComparisonVideo(t_rm, 'RM_diffusion_comparison.mp4', visual_style, steps = 200, initial_index=0)

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
visual_style["edge_width"] = [np.log(x)/2 for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_label"] = g1.vs["name"]
visual_style["edge_curved"] = .2
visual_style["edge_arrow_size"] = 0.5
visual_style["vertex_size"] = 20
visual_style["vertex_label_size"] = 10
visual_style["layout"] = g1.layout_kamada_kawai()
igraph.plot(g1, 'EM_g1.png', **visual_style)
igraph.plot(g1, 'EM_g1.pdf', **visual_style)

# Export diffusion videos
tn.exportDiffusionVideo(t_em, 'EM_diffusion_t2.mp4', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionVideo(t_em, 'EM_diffusion_t1.mp4', visual_style, steps = 200, initial_index=0, model='NULL')

# Plot temporal evolution
visual_style["vertex_color"] = "lightblue"
t_em.exportMovie('EM_temporal.mp4', visual_style, realtime=False)
t_em.exportMovie('EM_temporal_bursts.mp4', visual_style, realtime=True)

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
t_sd = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/sigma-0_75.trigram', sep= ' ', fformat='TRIGRAM')

# Plot second-order network
g2_su = t_su.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2_su.layout_kamada_kawai()
visual_style["edge_width"] = [(x*5)**2 for x in g2_su.es()["weight"]]
visual_style["edge_arrow_size"] = 0.2
visual_style["edge_curved"] = 0.25
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
igraph.plot(g2_su, 'Model_su_g2.pdf', **visual_style)

g2_sd = t_sd.igraphSecondOrder()
visual_style["edge_width"] = [(x*5)**1.5 for x in g2_sd.es()["weight"]]
igraph.plot(g2_sd, 'Model_sd_g2.pdf', **visual_style)

g2_sd_null = t_sd.igraphSecondOrderNull()
visual_style["edge_width"] = [x*300 for x in g2_sd_null.es()["weight"]]
igraph.plot(g2_sd_null, 'Model_null_g2.pdf', **visual_style)


# Plot first-order network
g1 = t_su.igraphFirstOrder()

visual_style = {}
visual_style["edge_width"] = [x for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_kamada_kawai()
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 20
visual_style["edge_curved"] = .25
igraph.plot(g1.as_undirected(), 'Model_g1.pdf', **visual_style)

# Export diffusion videos
tn.exportDiffusionVideo(t_su, 'Model_su_diffusion.mp4', visual_style, steps = 200, initial_index=1)
tn.exportDiffusionVideo(t_sd, 'Model_sd_diffusion.mp4', visual_style, steps = 200, initial_index=1)
tn.exportDiffusionVideo(t_sd, 'Model_null_diffusion.mp4', visual_style, steps = 200, initial_index=75, model='NULL')

tn.exportDiffusionComparisonVideo(t_su, 'Model_su_diffusion.mp4', visual_style, steps = 200, initial_index=50)
tn.exportDiffusionComparisonVideo(t_sd, 'Model_sd_diffusion.mp4', visual_style, steps = 200, initial_index=50)

# Spectral analysis
print('lambda_2(L_su) =', tn.AlgebraicConn(t_su))
print('lambda_2(L_sd) =', tn.AlgebraicConn(t_sd))

plotFiedler(t_su, 'Model_su_fiedler.pdf', col='g')
plotFiedler(t_su, 'Model_null_fiedler.pdf', model='NULL', col='b')
plotFiedler(t_sd, 'Model_sd_fiedler.pdf', col='r')

plotSpectrum(t_su, 'Model_su_spectrum.pdf', col='r')


# Spectral analysis
print('lambda_2(L) =', tn.AlgebraicConn(t_sd))
print('lambda_2(L_null) =', tn.AlgebraicConn(t_sd, model='NULL'))


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
