import pyTempNet as tn
import igraph 
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

t = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/tube_flows_scc.tedges', sep= ' ')
t.extractTwoPaths()

g2 = t.igraphSecondOrder()
visual_style = {}
visual_style["edge_width"] = [x**.1 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
visual_style["layout"] = g2.layout_auto()
igraph.plot(g2, 'LT_g2.pdf', **visual_style)

g1 = t.igraphFirstOrder()

visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'LT_g1.pdf', **visual_style)

tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/LT_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/LT_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

g2n = t.igraphSecondOrderNull()
visual_style["edge_width"] = [x**.1 for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.4
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
igraph.plot(g2n, 'LT_g2n.pdf', **visual_style)

generate_rw_frames(g2n, 'frames/LT_g2n', visual_style, steps = 1000)

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

# Do the same for Reality Mining 
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


# Do the same for the ants
t = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/ants-1-1_agg_6s_scc.tedges', sep= ' ')

t.extractTwoPaths()

g2 = t.igraphSecondOrder()
visual_style = {}
visual_style["layout"] = g2.layout_auto()

visual_style["edge_width"] = [x**.5 for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.2
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
igraph.plot(g2, 'AN_g2.pdf', **visual_style)

g1 = t.igraphFirstOrder()

visual_style["edge_width"] = [np.log(x)/4 for x in g1.es()["weight"]]
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'AN_g1.pdf', **visual_style)

tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/AN_diffusion_t2', visual_style, steps = 200, initial_index=0, model='SECOND')
tn.exportDiffusionMovieFramesFirstOrder(t, 'frames/AN_diffusion_t1', visual_style, steps = 200, initial_index=0, model='NULL')

# Do the same for the E-mail network
t = tn.TemporalNetwork.readFile('../../pyTempNetDemo/data/manufacturing_30d_agg_3600_scc.tedges', sep= ' ')

t.extractTwoPaths()

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





