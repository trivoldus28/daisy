from __future__ import absolute_import
from ..coordinate import Coordinate
from ..roi import Roi
from .shared_graph_provider import\
    SharedGraphProvider, SharedSubGraph
from ..graph import Graph, DiGraph
import itertools
import json
import logging
import numpy as np
import os
import shutil

logger = logging.getLogger(__name__)


class FileGraphProvider(SharedGraphProvider):
    '''Provides shared graphs stored in files.

    Nodes are assumed to have at least an attribute ``id`` and a position
    attribute (set via argument ``position_attribute``, defaults to
    ``position``).

    Edges are assumed to have at least attributes ``u``, ``v``.

    Arguments:

        directory (``string``):

            The path to the graph container directory.

        chunk_size (list-like of ``int``s):

            Chunk size of each file in the container.

        mode (``string``, optional):

            One of ``r``, ``r+``, or ``w``. Defaults to ``r+``. ``w`` drops the
            node, edge, and meta collections.

        directed (``bool``):

            True if the graph is directed, false otherwise

        roi_offset (list-like of ``int``s):

            Offset of the first chunk in the container. Useful when ROI
            does not align to `chunk_size`.

        nodes_collection (``string``):
        edges_collection (``string``):

            Names of the nodes and edges collections, should they differ from
            ``nodes`` and ``edges``.

        position_attribute (``string`` or list of ``string``s, optional):

            The node attribute(s) that contain position information. This will
            be used for slicing subgraphs via ``__getitem__``. If a single
            string, the attribute is assumed to be an array. If a list, each
            entry denotes the position coordinates in order (e.g.,
            `position_z`, `position_y`, `position_x`).

        save_attributes_as_single_file (``bool``, default ``False``):

            Whether to save attributes as a single file to increase storage
            efficiency or as separate files one for each attribute.

        nodes_chunk_size (list-like of ``int``s):
        edges_chunk_size (list-like of ``int``s):

            Allowing chunk size to be set separately for nodes and edges.

        nodes_roi_offset (list-like of ``int``s):
        edges_roi_offset (list-like of ``int``s):

            Allowing offset to be set separately for nodes and edges.

        nodes_no_misaligned_reads (``bool``, default ``False``):
        nodes_no_misaligned_writes (``bool``, default ``False``):
        edges_no_misaligned_reads (``bool``, default ``False``):
        edges_no_misaligned_writes (``bool``, default ``False``):

            Disallow misaligned chunk access. Useful as error assertions.

        nodes_no_filter_misaligned_reads (``bool``, default ``False``):

            When reads are misaligned, return all nodes within the chunk
            instead of filtering out nodes outside of the requested ROI.
    '''

    def __init__(
            self,
            directory,
            chunk_size,
            mode='r+',
            directed=None,
            total_roi=None,
            roi_offset=None,
            nodes_collection='nodes',
            edges_collection='edges',
            position_attribute='position',
            save_attributes_as_single_file=False,
            nodes_chunk_size=None,
            edges_chunk_size=None,
            nodes_roi_offset=None,
            edges_roi_offset=None,
            nodes_no_misaligned_reads=False,
            nodes_no_misaligned_writes=False,
            edges_no_misaligned_reads=False,
            edges_no_misaligned_writes=False,
            nodes_no_filter_misaligned_reads=False,
            ):

        self.directory = directory
        self.chunk_size = Coordinate(chunk_size)
        self.mode = mode
        self.directed = directed
        self.total_roi = total_roi
        self.nodes_collection_name = nodes_collection
        self.edges_collection_name = edges_collection
        self.position_attribute = position_attribute
        self.save_attributes_as_single_file = save_attributes_as_single_file
        self.nodes_no_misaligned_reads = nodes_no_misaligned_reads
        self.nodes_no_misaligned_writes = nodes_no_misaligned_writes
        self.edges_no_misaligned_reads = edges_no_misaligned_reads
        self.edges_no_misaligned_writes = edges_no_misaligned_writes
        self.nodes_no_filter_misaligned_reads = nodes_no_filter_misaligned_reads

        if roi_offset is None:
            roi_offset = ((0,)*len(chunk_size))
        self.roi_offset = Coordinate(roi_offset)

        self.nodes_chunk_size = chunk_size
        self.edges_chunk_size = chunk_size
        if nodes_chunk_size is not None:
            self.nodes_chunk_size = nodes_chunk_size
        if edges_chunk_size is not None:
            self.edges_chunk_size = edges_chunk_size
        self.nodes_chunk_size = Coordinate(self.nodes_chunk_size)
        self.edges_chunk_size = Coordinate(self.edges_chunk_size)

        self.nodes_roi_offset = roi_offset
        self.edges_roi_offset = roi_offset
        if nodes_roi_offset is not None:
            self.nodes_roi_offset = nodes_roi_offset
        if edges_roi_offset is not None:
            self.edges_roi_offset = edges_roi_offset
        self.nodes_roi_offset = Coordinate(self.nodes_roi_offset)
        self.edges_roi_offset = Coordinate(self.edges_roi_offset)

        self.nodes_collection = os.path.join(
            self.directory,
            self.nodes_collection_name)
        self.edges_collection = os.path.join(
            self.directory,
            self.edges_collection_name)
        self.meta_collection = os.path.join(
            self.directory,
            '.meta.json')

        if mode == 'w':

            logger.info(
                "dropping collections %s, %s",
                self.nodes_collection_name,
                self.edges_collection_name)

            shutil.rmtree(self.nodes_collection, ignore_errors=True)
            shutil.rmtree(self.edges_collection, ignore_errors=True)
            try:
                os.remove(self.meta_collection)
            except Exception:
                pass

        os.makedirs(self.nodes_collection, exist_ok=True)
        os.makedirs(self.edges_collection, exist_ok=True)

        if os.path.exists(self.meta_collection):
            self.__check_metadata()
        else:
            self.__set_metadata()

    def get_chunks(self, roi, chunk_size, roi_offset):
        '''Get a list of chunk indices and a list of chunk ROIs for each chunk
        that overlaps with the given ROI.'''

        roi = roi.shift(-roi_offset)
        chunk_roi = roi.snap_to_grid(chunk_size, mode='grow')
        chunks = chunk_roi/chunk_size

        chunk_indices = itertools.product(*[
            range(chunks.get_begin()[d], chunks.get_end()[d])
            for d in range(chunks.dims)
        ])

        return chunk_indices

    def __chunk_nodes_path(self, chunk_index):
        return os.path.join(
            self.nodes_collection,
            *[str(i) for i in chunk_index])

    def __chunk_edges_path(self, chunk_index):
        return os.path.join(
            self.edges_collection,
            *[str(i) for i in chunk_index])

    def __get_roi_filter(self, nodes, roi):

        if type(self.position_attribute) == list:
            num_nodes = len(nodes[self.position_attribute[0]])
            roi_filter = np.ones((num_nodes,), dtype=np.bool)
            for d in range(roi.dims):
                node_dim_values = nodes[self.position_attribute[d]]
                ge = np.array([node_value >= roi.get_begin()[d]
                               for node_value in node_dim_values])
                lt = np.array([node_value < roi.get_end()[d]
                               for node_value in node_dim_values])
                roi_filter &= (ge & lt)

        else:
            node_positions = nodes[self.position_attribute]
            num_nodes = len(node_positions)
            roi_filter = np.ones((num_nodes,), dtype=np.bool)
            for d in range(roi.dims):
                ge = np.array([pos[d] >= roi.get_begin()[d]
                               for pos in node_positions])
                lt = np.array([pos[d] < roi.get_end()[d]
                               for pos in node_positions])
                roi_filter &= (ge & lt)

        return roi_filter

    def _write_nodes_to_chunk(self, chunk_index, nodes, roi=None):

        chunk_roi = Roi(chunk_index, (1,)*self.nodes_chunk_size.dims)
        chunk_roi *= self.nodes_chunk_size
        chunk_roi += self.nodes_roi_offset

        path = self.__chunk_nodes_path(chunk_index)
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, '.meta.json'), 'w') as f:
            attributes = list(nodes.keys())
            json.dump({'attributes': attributes}, f)

        if roi is not None and not roi.contains(chunk_roi) and \
                not chunk_roi.contains(roi):

            roi_filter = self.__get_roi_filter(nodes, roi)
            for k, v in nodes.items():
                nodes[k] = list(np.array(nodes[k])[roi_filter])

        if self.save_attributes_as_single_file:
            np.savez_compressed(os.path.join(path, 'nodes'), **nodes)
        else:
            for k, v in nodes.items():
                np.savez_compressed(os.path.join(path, k), nodes=v)

    def _write_edges_to_chunk(
            self,
            chunk_index,
            edges,
            edge_positions,
            roi=None):

        chunk_roi = Roi(chunk_index, (1,)*self.edges_chunk_size.dims)
        chunk_roi *= self.edges_chunk_size
        chunk_roi += self.edges_roi_offset

        path = self.__chunk_edges_path(chunk_index)
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, '.meta.json'), 'w') as f:
            attributes = list(edges.keys())
            json.dump({'attributes': attributes}, f)

        if roi is not None and not roi.contains(chunk_roi) and \
                not chunk_roi.contains(roi):

            roi_filter = np.ones((len(edges),), dtype=np.bool)
            for d in range(roi.dims):
                ge = edge_positions[d] >= roi.get_begin()[d]
                lt = edge_positions[d] < roi.get_end()[d]
                roi_filter &= (ge & lt)

            for k, v in edges.items():
                edges[k] = edges[k][roi_filter]

        if self.save_attributes_as_single_file:
            np.savez_compressed(os.path.join(path, 'edges'), **edges)
        else:
            for k, v in edges.items():
                np.savez_compressed(os.path.join(path, k), edges=v)

    def _read_nodes_from_chunk(self, chunk_index, roi=None):

        chunk_roi = Roi(chunk_index, (1,)*self.nodes_chunk_size.dims)
        chunk_roi *= self.nodes_chunk_size
        chunk_roi += self.nodes_roi_offset

        path = self.__chunk_nodes_path(chunk_index)
        if not os.path.exists(path):
            return {'id': []}

        with open(os.path.join(path, '.meta.json'), 'r') as f:
            meta = json.load(f)

        nodes = {}

        if self.save_attributes_as_single_file:
            nodes_npz = np.load(os.path.join(path, 'nodes.npz'),
                                allow_pickle=True)
            for attribute in meta['attributes']:
                nodes[attribute] = nodes_npz[attribute]

        for attribute in meta['attributes']:
            file_path = os.path.join(path, attribute + '.npz')
            if os.path.exists(file_path):
                nodes[attribute] = np.load(file_path,
                                           allow_pickle=True)['nodes']

        if roi is None or roi.contains(chunk_roi) or \
                self.nodes_no_filter_misaligned_reads:
            return nodes

        roi_filter = self.__get_roi_filter(nodes, roi)
        for k, v in nodes.items():
            nodes[k] = nodes[k][roi_filter]
        return nodes

    def _read_edges_from_chunk(self, chunk_index, roi=None, node_ids=None):

        chunk_roi = Roi(chunk_index, (1,)*self.edges_chunk_size.dims)
        chunk_roi *= self.edges_chunk_size
        chunk_roi += self.edges_roi_offset

        path = self.__chunk_edges_path(chunk_index)
        if not os.path.exists(path):
            return {'u': [], 'v': []}

        with open(os.path.join(path, '.meta.json'), 'r') as f:
            meta = json.load(f)

        edges = {}

        if self.save_attributes_as_single_file:
            edges_npz = np.load(os.path.join(path, 'edges.npz'),
                                allow_pickle=True)
            for attribute in meta['attributes']:
                edges[attribute] = edges_npz[attribute]

        else:
            for attribute in meta['attributes']:
                file_path = os.path.join(path, attribute + '.npz')
                if os.path.exists(file_path):
                    edges[attribute] = np.load(file_path,
                                               allow_pickle=True)['edges']

        # we assume that if the chunk is contained in ROI, there is no need to
        # filter for node_ids any more
        if roi is None or roi.contains(chunk_roi):
            return edges

        roi_filter = np.isin(edges['u'], node_ids)
        for k, v in edges.items():
            edges[k] = edges[k][roi_filter]

        return edges

    def read_nodes(self, roi):
        '''Return a list of nodes within roi.'''

        nodes = {}
        logger.debug("Reading nodes in roi %s" % roi)

        if self.nodes_no_misaligned_reads:
            assert roi.shift(-self.nodes_roi_offset).begin.is_multiple_of(
                        self.nodes_chunk_size), "Misaligned nodes read " \
                        f"to {roi} with chunks {self.nodes_chunk_size}"

        for chunk_index in self.get_chunks(roi,
                                           self.nodes_chunk_size,
                                           self.nodes_roi_offset):
            logger.debug("Reading nodes in chunk %s" % str(chunk_index))
            chunk_nodes = self._read_nodes_from_chunk(chunk_index, roi)
            if len(chunk_nodes) == 0 or len(chunk_nodes['id']) == 0:
                logger.debug("Chunk %s and roi %s did not contain any nodes"
                             % (str(chunk_index), roi))
                continue
            for k, v in chunk_nodes.items():
                if k not in nodes:
                    nodes[k] = []
                nodes[k].append(v)

        for k, v in nodes.items():
            nodes[k] = np.concatenate(v)

        return nodes

    def num_nodes(self, roi):
        '''Return the number of nodes in the roi.'''

        # TODO: can be made more efficient
        return len(self.read_nodes(roi))

    def has_edges(self, roi):
        '''Returns true if there is at least one edge in the roi.'''

        # TODO: can be made more efficient
        return len(self.read_edges(roi)) > 0

    def read_edges(self, roi, nodes=None):
        '''Returns a list of edges within roi.'''

        if self.edges_no_misaligned_reads:
            assert roi.shift(-self.edges_roi_offset).begin.is_multiple_of(
                        self.edges_chunk_size), "Misaligned edges read " \
                        f"to {roi} with chunks {self.edges_chunk_size}"

        if nodes is None:
            nodes = self.read_nodes(roi)

        if len(nodes) == 0:
            return {}

        edges = {}
        for chunk_index in self.get_chunks(roi,
                                           self.edges_chunk_size,
                                           self.edges_roi_offset):

            chunk_edges = self._read_edges_from_chunk(
                chunk_index,
                roi,
                nodes['id'])
            if len(chunk_edges) == 0 or len(chunk_edges['u']) == 0:
                continue

            for k, v in chunk_edges.items():
                if k not in edges:
                    edges[k] = []
                edges[k].append(v)

        for k, v in edges.items():
            edges[k] = np.concatenate(v)

        return edges

    def __getitem__(self, roi):

        return self.get_graph(roi, empty_init=False, no_edges_init=False)

    def get_graph(
            self,
            roi,
            empty_init=False,
            no_edges_init=False):

        if self.directed:
            graph = FileSubDiGraph(
                self, roi,
                self.nodes_chunk_size, self.edges_chunk_size,
                self.nodes_roi_offset, self.edges_roi_offset,
                )
        else:
            graph = FileSubGraph(
                self, roi,
                self.nodes_chunk_size, self.edges_chunk_size,
                self.nodes_roi_offset, self.edges_roi_offset,
                )

        if empty_init:
            return graph

        # get all nodes within roi
        nodes = self.read_nodes(roi)

        if len(nodes) > 0:
            node_list = [
                (
                    nodes['id'][i],
                    {
                        k: v[i]
                        for k, v in nodes.items()
                        if k != 'id'
                    }
                )
                for i in range(len(nodes['id']))
            ]
            graph.add_nodes_from(node_list)

        if no_edges_init:
            return graph

        edges = self.read_edges(roi, nodes)

        if len(edges) > 0:
            edge_list = [
                (
                    edges['u'][i],
                    edges['v'][i],
                    {
                        k: v[i]
                        for k, v in edges.items()
                        if k != 'u' and k != 'v'
                    }
                )
                for i in range(len(edges['u']))
            ]
            graph.add_edges_from(edge_list)

        return graph

    def __get_metadata(self):
        '''Gets metadata out of the meta collection and returns it
        as a dictionary.'''

        with open(self.meta_collection, 'r') as f:
            return json.load(f)

    def __check_metadata(self):
        '''Checks if the provided metadata matches the existing
        metadata in the meta collection'''

        metadata = self.__get_metadata()
        if self.directed is not None and metadata['directed'] != self.directed:
            raise ValueError((
                    "Input parameter directed={} does not match"
                    "directed value {} already in stored metadata")
                    .format(self.directed, metadata['directed']))
        if self.total_roi:
            if self.total_roi.get_offset() != metadata['total_roi_offset']:
                raise ValueError((
                    "Input total_roi offset {} does not match"
                    "total_roi offset {} already stored in metadata")
                    .format(
                        self.total_roi.get_offset(),
                        metadata['total_roi_offset']))
            if self.total_roi.get_shape() != metadata['total_roi_shape']:
                raise ValueError((
                    "Input total_roi shape {} does not match"
                    "total_roi shape {} already stored in metadata")
                    .format(
                        self.total_roi.get_shape(),
                        metadata['total_roi_shape']))

    def __set_metadata(self):
        '''Sets the metadata in the meta collection to the provided values'''

        if not self.directed:
            # default is false
            self.directed = False
        if not self.total_roi:
            # default is an unbounded roi
            self.total_roi = Roi((0, 0, 0, 0), (None, None, None, None))

        meta_data = {
                'directed': self.directed,
                'total_roi_offset': self.total_roi.get_offset(),
                'total_roi_shape': self.total_roi.get_shape()
            }

        with open(self.meta_collection, 'w') as f:
            json.dump(meta_data, f)


class FileSharedSubGraph(SharedSubGraph):

    def __init__(
            self,
            graph_provider,
            roi,
            nodes_chunk_size,
            edges_chunk_size,
            nodes_roi_offset,
            edges_roi_offset
            ):

        super().__init__()

        self.graph_provider = graph_provider
        self.roi = roi
        self.pos_list = type(self.graph_provider.position_attribute) == list
        self.nodes_chunk_size = nodes_chunk_size
        self.edges_chunk_size = edges_chunk_size
        self.nodes_roi_offset = nodes_roi_offset
        self.edges_roi_offset = edges_roi_offset

    def __get_node_pos(self, n):

        try:

            if self.pos_list:

                return Coordinate((
                    n[pos_attr]
                    for pos_attr in self.graph_provider.position_attribute))

            else:

                return Coordinate(
                    n[self.graph_provider.position_attribute])

        except KeyError:

            return None

    def write_edges(
            self,
            roi=None,
            attributes=None,
            fail_if_exists=False,
            fail_if_not_exists=False,
            delete=False):
        assert not delete, "Delete not implemented"
        assert not(fail_if_exists and fail_if_not_exists),\
            "Cannot have fail_if_exists and fail_if_not_exists simultaneously"
        if fail_if_exists:
            raise RuntimeError("Fail if exists not implemented for "
                               "file backend")
        if fail_if_not_exists:
            raise RuntimeError("Fail if not exists not implemented for "
                               "file backend")
        if attributes is not None:
            raise RuntimeError("Attributes not implemented for file backend")
        if self.graph_provider.mode == 'r':
            raise RuntimeError("Trying to write to read-only DB")
        if roi is None:
            roi = self.roi

        logger.debug("Writing edges in %s", roi)

        if self.graph_provider.edges_no_misaligned_writes:
            assert roi.shift(-self.edges_roi_offset).begin.is_multiple_of(
                        self.edges_chunk_size), "Misaligned edges write " \
                        f"to {roi} with chunks {self.edges_chunk_size}"

        edges = {'u': [], 'v': []}
        edge_positions = []
        for u, v, data in self.edges(data=True):
            if not self.is_directed():
                u, v = min(u, v), max(u, v)
            pos = self.__get_node_pos(self.nodes[u])
            if pos is None or not roi.contains(pos):
                continue
            edges['u'].append(np.uint64(u))
            edges['v'].append(np.uint64(v))
            edge_positions.append(pos)
            for k, v in data.items():
                if k not in edges:
                    num_entries = len(edges['u'])
                    edges[k] = [None]*(num_entries - 1)
                edges[k].append(v)

        num_entries = len(edges['u'])

        if num_entries == 0:
            logger.debug("No edges to insert in %s", roi)
            return

        for k, v in edges.items():
            v += [None]*(num_entries - len(v))

        for chunk in self.graph_provider.get_chunks(roi,
                                                    self.edges_chunk_size,
                                                    self.edges_roi_offset):
            self.graph_provider._write_edges_to_chunk(chunk, edges, roi)

    def write_nodes(
            self,
            roi=None,
            attributes=None,
            fail_if_exists=False,
            fail_if_not_exists=False,
            delete=False):
        assert not delete, "Delete not implemented"
        assert not(fail_if_exists and fail_if_not_exists),\
            "Cannot have fail_if_exists and fail_if_not_exists simultaneously"
        if fail_if_exists:
            raise RuntimeError("Fail if exists not implemented for "
                               "file backend")
        if fail_if_not_exists:
            raise RuntimeError("Fail if not exists not implemented for "
                               "file backend")
        if attributes is not None:
            raise RuntimeError("Attributes not implemented for file backend")
        if self.graph_provider.mode == 'r':
            raise RuntimeError("Trying to write to read-only DB")

        if roi is None:
            roi = self.roi

        logger.debug("Writing nodes in %s", roi)

        if self.graph_provider.nodes_no_misaligned_writes:
            assert roi.shift(-self.nodes_roi_offset).begin.is_multiple_of(
                        self.nodes_chunk_size), "Misaligned nodes write " \
                        f"to {roi} with chunks {self.nodes_chunk_size}"

        nodes = {'id': []}
        for n, data in self.nodes(data=True):
            pos = self.__get_node_pos(self.nodes[n])
            if pos is None or not roi.contains(pos):
                continue
            nodes['id'].append(np.uint64(n))
            for k, v in data.items():
                if k not in nodes:
                    num_entries = len(nodes['id'])
                    nodes[k] = [None]*(num_entries - 1)
                nodes[k].append(v)

        num_entries = len(nodes['id'])
        logger.debug("ids have type %s" % type(nodes['id'][0]))

        if num_entries == 0:
            logger.debug("No nodes to insert in %s", roi)
            return

        for k, v in nodes.items():
            v += [None]*(num_entries - len(v))

        for chunk in self.graph_provider.get_chunks(roi,
                                                    self.nodes_chunk_size,
                                                    self.nodes_roi_offset):
            self.graph_provider._write_nodes_to_chunk(chunk, nodes, roi)

    def __contains(self, roi, node):
        '''Determines if the given node is inside the given roi'''
        node_data = self.node[node]

        # Some nodes are outside of the originally requested ROI (they have
        # been pulled in by edges leaving the ROI). These nodes have no
        # attributes, so we can't perform an inclusion test. However, we
        # know they are outside of the subgraph ROI, and therefore also
        # outside of 'roi', whatever it is.
        if 'position' not in node_data:
            return False

        return roi.contains(Coordinate(node_data['position']))

    def is_directed(self):
        raise RuntimeError("not implemented in %s" % self.name())


class FileSubGraph(FileSharedSubGraph, Graph):
    def __init__(
            self,
            graph_provider,
            roi,
            nodes_chunk_size,
            edges_chunk_size,
            nodes_roi_offset,
            edges_roi_offset):
        # this calls the init function of the FileSharedSubGraph,
        # because left parents come before right parents
        super().__init__(
                graph_provider,
                roi,
                nodes_chunk_size,
                edges_chunk_size,
                nodes_roi_offset,
                edges_roi_offset)

    def is_directed(self):
        return False


class FileSubDiGraph(FileSharedSubGraph, DiGraph):
    def __init__(
            self,
            graph_provider,
            roi,
            nodes_chunk_size,
            edges_chunk_size,
            nodes_roi_offset,
            edges_roi_offset):
        # this calls the init function of the FileSharedSubGraph,
        # because left parents come before right parents
        super().__init__(
                graph_provider,
                roi,
                nodes_chunk_size,
                edges_chunk_size,
                nodes_roi_offset,
                edges_roi_offset)

    def is_directed(self):
        return True
