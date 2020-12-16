import os
import gzip
import networkx as nx
import matplotlib.pyplot as plt

WIKI_GZ_FILE_PATH = 'PATH/TO/FILE/wiki-Vote.txt.gz'  # specify the location of the '.txt.gz' file
WIKI_TXT_FILE_PATH = '../data/wiki/wiki-Vote.txt'  # location (relative or absolute) of the decompressed (.txt) file


def decompress_wiki_file():
    if os.path.exists(WIKI_TXT_FILE_PATH):
        print(f'file already decompressed -> {WIKI_TXT_FILE_PATH}\n')
        return

    print('decompressing file...')
    with gzip.open(WIKI_GZ_FILE_PATH, 'rt') as original_file:
        with open(WIKI_TXT_FILE_PATH, 'w') as edge_list_file:
            for line in original_file.readlines():
                if line.startswith('#'):  # skipping the first few lines with metadata
                    continue
                edge_list_file.write(line)
    print(f'file decompressed -> {WIKI_TXT_FILE_PATH}\n')


def is_multi_graph():
    """
    In my understanding of the dataset, voters can vote only once for a candidate user. But what if a user was a
    candidate (nominated) multiple times? Having no information of the elections themselves (id), that would mean
    that in the dataset there would be two rows with the same values, ex: '2 3\n2 3', meaning that user '2' voted for
    user '3' two times on two different occasions (elections). This use-case is important because it decides if the
    graph should allow multiple edges going from one node to another (`networkx.MultiDiGraph`), or not allow
    (`networkx.DiGraph`). No information about this was given on the 'SNAP' website.

    Two possible ways to see if the graph needs to be `networkx.MultiDiGraph`: manually check in the dataset to see if
    there are multiple edges between two nodes, or write an algorithm to check. It's a little hard to manually check
    10000+ rows, so i wrote this algorithm.

    If we inspect the file we can notice that the voters are sorted in ascending order. Also, the votes (candidates) of
    the voters are also sorted in ascending order. If we imagine the dataset to be a list of tuples t, where t[0] is the
    voter and t[1] is the vote of the voter t[0], then the list would be sorted like so:
    `sorted(tuples, key=lambda t: (t[0], t[1]))`. So, if there are actually multiple votes by one voter to one candidate
    then in the list of tuples, those tuples would be next to each other. I utilize this characteristic of the dataset
    in this algorithm.
    """
    with open(WIKI_TXT_FILE_PATH, 'rt') as edges_file:
        prev_voter, prev_candidate = None, None
        for line in edges_file.readlines():
            parts = line.strip().split('\t')
            curr_voter, curr_candidate = int(parts[0]), int(parts[1])

            if prev_voter is None or prev_candidate is None:
                prev_voter, prev_candidate = curr_voter, curr_candidate
                continue

            if curr_voter == prev_voter and curr_candidate == prev_candidate:
                return True

            prev_voter, prev_candidate = curr_voter, curr_candidate
    return False


def plot_graph_degree_distribution(x, y, color, scale_x, scale_y, label_x, label_y, title):
    plt.plot(x, y, color)
    plt.xscale(scale_x)
    plt.yscale(scale_y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)


def plot_digraph_degree_distribution(props: dict):
    plt.subplot(211)
    plot_graph_degree_distribution(props['x1'], props['y1'], props['color1'], props['scale-x1'],
                                   props['scale-y1'], props['label-x1'], props['label-y1'], props['title1'])

    plt.subplot(212)
    plot_graph_degree_distribution(props['x2'], props['y2'], props['color2'], props['scale-x2'],
                                   props['scale-y2'], props['label-x2'], props['label-y2'], props['title2'])


def display_distributions(wiki_vote, gnp, watts, num_nodes):
    in_degrees = dict(wiki_vote.in_degree())
    in_deg_x = sorted(set(in_degrees.values()))  # set of all in-degrees in the graph to draw on the x-axis
    # in_deg_freq: key->in-deg; value->#nodes with 'key' in-deg
    in_deg_freq = dict([(deg, list(in_degrees.values()).count(deg)) for deg in in_deg_x])
    prob_in_deg_y = [in_deg_freq[deg] / num_nodes for deg in in_deg_x]

    out_degrees = dict(wiki_vote.out_degree())
    out_deg_x = sorted(set(out_degrees.values()))
    # out_deg_freq: key->out-deg; value->#nodes with 'key' out-deg
    out_deg_freq = dict([(deg, list(out_degrees.values()).count(deg)) for deg in out_deg_x])
    prob_out_deg_y = [out_deg_freq[deg] / num_nodes for deg in out_deg_x]

    plot_props = {
        'x1': in_deg_x, 'y1': prob_in_deg_y, 'color1': 'r', 'scale-x1': 'log', 'scale-y1': 'log',
        'label-x1': 'In-Degree', 'label-y1': 'Probability', 'title1': 'In-Degree distribution (log) -> WIKI_VOTE',
        'x2': out_deg_x, 'y2': prob_out_deg_y, 'color2': 'b', 'scale-x2': 'log', 'scale-y2': 'log',
        'label-x2': 'Out-Degree', 'label-y2': 'Probability', 'title2': 'Out-Degree distribution (log) -> WIKI_VOTE'
    }
    plot_digraph_degree_distribution(plot_props)
    plt.show()

    gnp_degrees = dict(nx.degree(gnp))
    watts_degrees = dict(nx.degree(watts))

    x = sorted(set(gnp_degrees.values()))
    freq = dict([(deg, list(gnp_degrees.values()).count(deg)) for deg in x])
    y = [freq[deg] / num_nodes for deg in x]
    # plotting distribution for 'gnp' graph
    plot_graph_degree_distribution(x, y, 'g', 'log', 'log', 'Degree', 'Probability',
                                   'Degree distribution (log) -> Random Graph')
    plt.show()

    x = sorted(set(watts_degrees.values()))
    freq = dict([(deg, list(watts_degrees.values()).count(deg)) for deg in x])
    y = [freq[deg] / num_nodes for deg in x]
    # plotting distribution for 'watts' graph
    plot_graph_degree_distribution(x, y, 'g', 'log', 'log', 'Degree', 'Probability',
                                   'Degree distribution (log) -> Small-World')
    plt.show()


if __name__ == '__main__':
    decompress_wiki_file()

    wiki_vote = None
    if is_multi_graph():
        print('Directed MultiGraph -> networkx.MultiDiGraph\n')
        wiki_vote = nx.read_edgelist(WIKI_TXT_FILE_PATH, create_using=nx.MultiDiGraph)
    else:
        print('Directed Graph -> networkx.DiGraph\n')
        wiki_vote = nx.read_edgelist(WIKI_TXT_FILE_PATH, create_using=nx.DiGraph)

    num_nodes = nx.number_of_nodes(wiki_vote)

    gnp = nx.gnp_random_graph(num_nodes, 0.1)
    watts = nx.watts_strogatz_graph(num_nodes, 10, 0.1)

    print('# NODES')
    print(f'WIKI-VOTE: {num_nodes}')
    print(f'Random Graph: {num_nodes}')
    print(f'Small-World: {num_nodes}\n')

    print('# EDGES')
    print(f'WIKI-VOTE: {nx.number_of_edges(wiki_vote)}')
    print(f'Random Graph: {nx.number_of_edges(gnp)}')
    print(f'Small-World: {nx.number_of_edges(watts)}\n')

    display_distributions(wiki_vote, gnp, watts, num_nodes)

    # connected components
    # because wiki-vote is a directed graph, we have strongly connected components and weakly connected components
    num_wcc = nx.number_weakly_connected_components(wiki_vote)
    num_scc = nx.number_strongly_connected_components(wiki_vote)
    gnp_num_cc = nx.number_connected_components(gnp)
    watts_num_cc = nx.number_connected_components(watts)

    print('CONNECTED COMPONENTS (CC)')
    print(f'WIKI-VOTE: # Weakly CC: {num_wcc} {"-> IS STRONGLY CONNECTED" if num_scc == 1 else ""},'
          f' # Strongly CC: {num_scc} {"-> IS WEAKLY CONNECTED" if num_wcc == 1 else ""}')
    print(f'Random Graph: # CC: {gnp_num_cc} {"-> IS CONNECTED" if gnp_num_cc == 1 else ""}')
    print(f'Small-World: # CC: {watts_num_cc} {"-> IS CONNECTED" if watts_num_cc == 1 else ""}\n')

    wcc = nx.weakly_connected_components(wiki_vote)
    max_wcc = max(wcc, key=len)
    scc = nx.strongly_connected_components(wiki_vote)
    max_scc = max(scc, key=len)

    gnp_cc = nx.connected_components(gnp)
    max_gnp_cc = max(gnp_cc, key=len)

    watts_cc = nx.connected_components(watts)
    max_watts_cc = max(watts_cc, key=len)

    print('LARGEST CONNECTED COMPONENT (# NODES)')
    print(f'WIKI-VOTE: WCC: {len(max_wcc)}, SCC: {len(max_scc)}')
    print(f'Random Graph: {len(max_gnp_cc)}')
    print(f'Small-World: {len(max_watts_cc)}\n')

    print('DIAMETER')
    # we can calculate the diameter only of a connected (strongly) graph
    # if the graph is not connected (strongly), we calculate the diameter of the largest connected (strongly) component
    print(f'WIKI-VOTE:{nx.diameter(wiki_vote) if num_scc == 1 else nx.diameter(nx.subgraph(wiki_vote, max_scc))}')
    print(f'Random Graph:{nx.diameter(gnp) if gnp_num_cc == 1 else nx.diameter(nx.subgraph(gnp, max_gnp_cc))}')
    print(f'Small-World:{nx.diameter(watts) if watts_num_cc == 1 else nx.diameter(nx.subgraph(watts, max_watts_cc))}\n')

    print('AVERAGE CLUSTERING COEFFICIENT')
    print(f'WIKI-VOTE: {nx.average_clustering(wiki_vote)}')
    print(f'Random Graph: {nx.average_clustering(gnp)}')
    print(f'Small-World: {nx.average_clustering(watts)}\n')
