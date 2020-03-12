import itertools, re, os, tempfile, operator, functools, sys
import pandas as pd
import numpy as np
import ptnet
import prince

from hashlib import sha512
from IPython.display import display
import bqplot as bq
import ipywidgets as ipw
import networkx as nx

from .. import Model as _Model, CompileError, Record, help, cached_property
from .st import Parser, FailedParse, State
from ..graphs import Graph, Palette, to_graph
from ..ui import log, getopt, updated, HTML
from .states import load
from ..tables import read_csv, write_csv
from .statexpr import expr2sdd

def _set2str (s) :
    return "|".join(str(v) for v in sorted(s))

def parse (path) :
    try :
        return Parser(path).parse()
    except FailedParse as err :
        raise CompileError("\n".join(str(err).splitlines()[:3]))

class TableProxy (Record) :
    _fields = ["table", "columns"]
    _name = re.compile("^[a-z][a-z0-9_]*$", re.I)
    def __init__ (self, table) :
        super().__init__(table, list(table.columns))
    def __getitem__ (self, key) :
        if isinstance(key, str) :
            key = [key]
        elif not isinstance(key, slice) :
            key = list(key)
        stats = self.table[key].describe(include="all").fillna("")
        stats[" "] = np.full(len(stats), "")
        for col, val in {"count" : "total number of values",
                         "unique" : "number of unique values",
                         "top" : "most common value",
                         "freq" : "top's frequency",
                         "mean" : "values average",
                         "std" : "standard deviation",
                         "min" : "minimal value",
                         "max" : "maximal value",
                         "25%" : "25th percentile",
                         "50%" : "50th percentile",
                         "75%" : "75th percentile"}.items() :
            if col in stats.columns :
                stats.loc[col," "] = val
        return stats
    def __delitem__ (self, col) :
        if col in self.columns :
            display(HTML('<span style="color:#800; font-weight:bold;">warning:</span>'
                         ' you should not delete column %r,'
                         ' but if you insist I\'ll do it.' % col))
            self.columns.remove(col)
        elif isinstance(col, str) and col in self.table.columns :
            self.table.drop(columns=[col], inplace=True)
        elif isinstance(col, tuple) :
            for c in col :
                del self[c]
        else :
            raise KeyError("invalid column: %r" % (col,))
    def __setitem__ (self, key, val) :
        if key in self.table.columns :
            raise KeyError("columns %r exists already" % key)
        if callable(val) :
            data = self.table.apply(val, axis=1)
        elif isinstance(val, tuple) and len(val) > 1 and callable(val[0]) :
            fun, *args = val
            data = self.table.apply(fun, axis=1, args=args)
        else :
            data = pd.DataFrame({"col" : val})
        if isinstance(key, (tuple, list)) :
            for k in key :
                if not self._name.match(k) :
                    raise ValueError("invalid column name %r" % k)
            data = pd.DataFrame.from_records(data)
            for k, c in zip(key, data.columns) :
                self.table[k] = data[c]
        else :
            if not self._name.match(key) :
                raise ValueError("invalid column name %r" % key)
            self.table[key] = data
    def _ipython_display_ (self) :
        display(self[:])

class _View (Record) :
    def _path (self, *args) :
        if len(args) == 0 :
            return self.base_path / self.name
        elif len(args) == 1 :
            parts, ext = (), args[0]
        else :
            *parts, ext = args
        if not ext.startswith(".") :
            ext = "." + ext
        if parts :
            return (self.base_path / "-".join(str(p) for p in parts)).with_suffix(ext)
        else :
            return (self.base_path / self.name).with_suffix(ext)

@help
class ExplicitView (_View) :
    """a state-oriented view of a model (aka, a statespace)
    Attributes:
     - name: unique name of this view
     - base_path: where data is saved path
    """
    _fields = ["name", "base_path"]
    def __init__ (self, view, components) :
        self.components = [view[c] for c in sorted(components)]
        name = "explicit-" + "-".join(str(c.n) for c in self.components)
        super().__init__(name=name,
                         base_path=view.base_path / name)
        if not self.base_path.exists() :
            self.base_path.mkdir(parents=True)
        self.v = view
        self.g = view.g
        self.m = view.m
    def build (self, force=False) :
        if force or updated([self.v._path("ddd")],
                            [self._path("nodes", ".csv.bz2"),
                             self._path("edges", ".csv.bz2")]) :
            self.save()
    def save (self) :
        "save the nodes and edges tables to CSV"
        self.g.dump_x(self.components,
                      str(self._path("nodes", ".csv.bz2")),
                      str(self._path("edges", ".csv.bz2")))
    @cached_property
    def nodes (self) :
        return read_csv(self._path("nodes", ".csv.bz2"), state=self.m.state)
    @cached_property
    def edges (self) :
        return read_csv(self._path("edges", ".csv.bz2"), state=self.m.state)
    @cached_property
    def n (self) :
        return TableProxy(self.nodes)
    @cached_property
    def e (self) :
        return TableProxy(self.edges)
    def draw (self, **opt) :
        """draw the view (component graph)
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (600): figure height (in pixels)
         - fig_padding (0.01): internal figure margins
         - fig_title (None): figure title
        Graph options:
         - graph_layout ("fdp"): layout engine to compute nodes positions
        Nodes options:
         - nodes_label ("node"): column that defines nodes labels
         - nodes_shape (auto): column that defines nodes shape
         - nodes_size (35): nodes width
         - nodes_color ("component"): column that defines nodes colors
         - nodes_selected ("#EE0000"): color of the nodes selected for inspection
         - nodes_palette ("Pastel28"): palette for the nodes colors
         - nodes_ratio (1.2): height/width ratio of non-symmetrical nodes
        """
        opt.setdefault("graph_engines", {})["CA"] = self.ca()
        return Graph(self.nodes,
                     self.edges,
                     defaults={
                         "nodes_color" : "component",
                         "nodes_colorscale" : "discrete",
                         "nodes_shape" : (["init", "dead", "scc", "hull"],
                                          self._nodes_shape),
                         "marks_shape" : (["init", "dead", "scc", "hull"],
                                          self._marks_shape),
                         "marks_palette" : ["#FFFFFF", "#000000"],
                         "marks_opacity" : .5,
                         "marks_stroke" : "#888888",
                     }, **opt)
    def _nodes_shape (self, row) :
        if row.scc :
            return "circ"
        elif row.dead :
            return "sbox"
        else :
            return "rbox"
    def _marks_shape (self, row) :
        if row.init :
            return "triangle-down"
        elif row.hull :
            return "circle"
    def count (self) :
        """return a table with variable valuation for each state
        """
        variables = self.m.state.vars()
        return pd.DataFrame.from_records(
            [[int(v in s) for v in variables] for s in self.nodes["on"]],
            index=self.nodes.index,
            columns=variables)
    def ca (self,
            n_components=2, n_iter=3, copy=True, check_input=True,
            engine="auto", random_state=42) :
        """correspondence analysis of the count matrix

        See https://github.com/MaxHalford/prince#correspondence-analysis-ca
        for documentation about the options.
        """
        count = self.count()
        count = count[count.sum(axis="columns") > 0]
        ca = prince.CA(n_components=n_components,
                       n_iter=n_iter,
                       copy=copy,
                       check_input=check_input,
                       engine=engine,
                       random_state=random_state)
        ca.fit(count)
        trans = ca.transform(count)
        for idx in set(count.index) - set(trans.index) :
            trans.loc[idx] = [0, 0]
        return trans

@help
class ComponentView (_View) :
    """a component-oriented view of a model (aka, a component graph)
    Attributes:
     - name: unique name of this view
     - compact: whether transient states and constraints are hidden or not
     - nodes_count: number of nodes
     - edges_count: number of edges
     - rr_path: path of RR specification
    """
    _fields = ["name", "compact", "rr_path", "base_path"]
    def __init__ (self, name, model, **k) :
        self.model = model
        super().__init__(rr_path=model.path,
                         base_path=model.base / name,
                         name=name,
                         **k)
        self.base_path /= "compact" if self.compact else "expanded"
        if not self.base_path.exists() :
            self.base_path.mkdir(parents=True)
    def __getitem__ (self, key) :
        return self.g[key]
    def __delitem__ (self, key) :
        self.drop(self)
    def __len__ (self) :
        return self.g.nodes_count
    def __iter__ (self) :
        return iter(self.g)
    ##
    ## build and i/o
    ##
    def build (self, split: bool=True, force: bool=False, profile: bool=False) :
        self.model.compile(force, profile)
        mod = self.m = self.model.states
        if force or updated([self.rr_path], [self.model["gal"]]) :
            force = True
            self.model.gal()
        if (force
            or updated([self.rr_path], [self._path("ddd")])) :
            force = True
            self.g = mod.Graph(str(self.model["gal"]), self.compact, True)
            self.g.build(split)
            self.g.save(str(self._path("ddd")))
        else :
            self.g = mod.Graph.load(str(self._path("ddd")))
        self.nodes_count = self.g.nodes_count
        self.edges_count = self.g.edges_count
        if (force
            or updated([self.rr_path],
                       [self._path("nodes", "csv.bz2"),
                        self._path("edges", "csv.bz2")])) :
            self.g.to_csv(str(self._path("nodes", "csv.bz2")),
                          str(self._path("edges", "csv.bz2")))
        with log(head="<b>loading</b>",
                 done_head="<b>loaded:</b>",
                 tail="{step}",
                 done_tail = "%s nodes / %s edges" % (self.nodes_count,
                                                      self.edges_count),
                 steps=[str(self._path("nodes", "csv.bz2")),
                        str(self._path("edges", "csv.bz2"))],
                 keep=True) :
            self.nodes = read_csv(self._path("nodes", "csv.bz2"))
            self.n = TableProxy(self.nodes)
            log.update()
            self.edges = read_csv(self._path("edges", "csv.bz2"))
            self.e = TableProxy(self.edges)
            log.update()
    def save (self) :
        "save the nodes and edges tables to CSV"
        self.g.save()
        write_csv(self.nodes, self._path("nodes", "csv.bz2"), self.m.state)
        write_csv(self.edges, self._path("edges", "csv.bz2"), self.m.state)
    def load (self) :
        self.build(False, False)
    def explicit (self, *components, force : bool = False) -> ExplicitView :
        """build an explicit view of chosen components
        Options:
         - components (all): which components should be explicited
        """
        if not components :
            components = [c.n for c in self.g]
        dump = ExplicitView(self, components)
        dump.build(force=force)
        return dump
    ##
    ## draw
    ##
    def draw (self, **opt) :
        """draw the view (component graph)
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (600): figure height (in pixels)
         - fig_padding (0.01): internal figure margins
         - fig_title (None): figure title
        Graph options:
         - graph_layout ("fdp"): layout engine to compute nodes positions
        Nodes options:
         - nodes_label ("node"): column that defines nodes labels
         - nodes_shape (auto): column that defines nodes shape
         - nodes_size (35): nodes width
         - nodes_color ("size"): column that defines nodes colors
         - nodes_selected ("#EE0000"): color of the nodes selected for inspection
         - nodes_palette ("GRW"): palette for the nodes colors
         - nodes_ratio (1.2): height/width ratio of non-symmetrical nodes
        """
        opt.setdefault("graph_engines", {})["PCA"] = self.pca()
        return Graph(self.nodes,
                     self.edges,
                     defaults={
                         "nodes_color" : "size",
                         "nodes_colorscale" : "linear",
                         "nodes_palette" : "GRW",
                         "nodes_shape" : (["init", "dead", "scc", "hull"],
                                          self._nodes_shape),
                         "marks_shape" : (["init", "dead", "scc", "hull"],
                                          self._marks_shape),
                         "marks_palette" : ["#FFFFFF", "#000000"],
                         "marks_opacity" : .5,
                         "marks_stroke" : "#888888",
                     }, **opt)
    def _nodes_shape (self, row) :
        if row.scc :
            return "circ"
        elif row.dead :
            return "sbox"
        else :
            return "rbox"
    def _marks_shape (self, row) :
        if row.init :
            return "triangle-down"
        elif row.hull :
            return "circle"
    ##
    ## analyse components
    ##
    def count (self, *nums: int) -> pd.DataFrame :
        """count in how many states each variable is on in the components given
        as arguments
        Options:
         - nums (all): components numbers
        """
        if not nums :
            nums = [c.n for c in self.g]
        compo = [self.g[n] for n in nums]
        count = [c.count() for c in compo]
        return pd.DataFrame.from_records([[c for _, c in cnt] for cnt in count],
                                         index=[c.n for c in compo],
                                         columns=[v for v, _ in count[0]])
    def pca (self,
             n_components=2, n_iter=3, copy=True, check_input=True,
             engine="auto", random_state=42,
             rescale_with_mean=True, rescale_with_std=True) :
        """principal component analysis of the count matrix

        See https://github.com/MaxHalford/prince#principal-component-analysis-pca
        for documentation about the options.
        """
        count = self.count()
        count = count[count.sum(axis="columns") > 0]
        pca = prince.PCA(n_components=n_components,
                         n_iter=n_iter,
                         copy=copy,
                         check_input=check_input,
                         engine=engine,
                         random_state=random_state,
                         rescale_with_mean=rescale_with_mean,
                         rescale_with_std=rescale_with_std)
        pca.fit(count)
        trans = pca.transform(count)
        for idx in set(count.index) - set(trans.index) :
            trans.loc[idx] = [0, 0]
        return trans
    ##
    ## split components
    ##
    def _name2sdd (self, name: str) :
        if name in self.g.variables :
            return self.g[name]
        elif re.match("^[RC][0-9]+$", name) :
            rule = getattr(self.model.spec, "rules" if name[0] == "R"
                           else "constraints")[int(name[1:]) - 1]
            sdds = []
            for state in rule.left :
                if state.sign :
                    sdds.append(self.g[state.name])
                else :
                    sdds.append(self.g["*"] - self.g[state.name])
            return functools.reduce(operator.and_, sdds)
        elif name == "*" :
            return self.g["*"]
        else :
            raise ValueError("unknown variable or rule name %r" % name)
    def split (self, num: int, on: str="", normalise: bool=False, keeplog=True) :
        """split a component in two
        Arguments:
         - num: component number
        Options:
         - on: if not provided, the component is split into its SCC hull and the
           related components. Otherwise, `on` should be a state expression,
           that is a Boolean expression such that the atoms the atoms are:
            - a variable name which selects the states in which the variable is 1
            - a rule/constraint name which selects the states in which the rule is
              enabled (ie, may be executed)
           and the operations are:
            - `~expr` (NOT) which selects the states that are not selected
              by `expr`
            - `left | right` (OR) which selects the states that are in `left`,
              `right`, or both
            - `left & right` (AND) which selects the states that are both in
              `left` and `right`
            - `left ^ right` (XOR) which selects the states that are either in
              `left` or `right` but not in both
            - `(expr)` to group sub-expressions and enforce operators priorities
         - normalise (False): extract the SCC hull and related components from
           the components resulting from the split when `on` is provided
        """
        with log(head="splitting",
                 tail="{step} (TIME: {time} | MEM: {memory:.1f}%)",
                 done_head = "<b>split:</b>",
                 steps=["compiling expression", "splitting component",
                        "extracting SCC hulls", "updating tables"],
                 keep=keeplog) :
            if on :
                states = expr2sdd(on, self._name2sdd)
                log.update()
                patch = self.g.split_states(num, states)
                if not patch :
                    log.finish(done_tail="%s cannot be split (TIME: {time})" % num)
                    return
                log.update()
                # normalise result
                if normalise :
                    compo = list(patch)
                    for c in compo :
                        p = self.g.split_hull(c)
                        patch.components.extend(p.components[1:])
                        patch += p
            else :
                log.update(2)
                patch = self.g.split_hull(self.g[num])
                if not patch :
                    log.finish(done_tail="%s cannot be split (TIME: {time})" % num)
                    return
            log.update()
            self._patch_tables(patch)
            log.done_tail = "%s => %s (TIME: {time})" % (num, ", ".join(str(c.n)
                                                                        for c in patch))
            log.update()
    def _patch_tables (self, patch) :
        # update nodes table
        for compo in patch.components :
            on, off = compo.on_off()
            succ = self.g.getsucc(compo.n)
            pred = self.g.getpred(compo.n)
            row = {"node" : compo.n,
                   "size" : len(compo),
                   "succ" : "|".join(str(s) for s in sorted(succ)),
                   "pred" : "|".join(str(s) for s in sorted(pred)),
                   "on" : on,
                   "off" : off,
                   "init" : compo.init,
                   "dead" : compo.dead,
                   "scc" : compo.scc,
                   "hull" : compo.hull}
            found = self.nodes[self.nodes["node"] == compo.n]
            if len(found) :
                idx = found.index[0]
            else :
                idx = self.nodes.index.max() + 1
            self.nodes.loc[idx] = [row.get(c) for c in self.nodes.columns]
        self.nodes_count = self.g.nodes_count
        # update edges table
        edges = self.edges[["src", "dst"]].apply(tuple, axis="columns")
        self.edges.drop(index=self.edges[edges.isin(patch.rem)].index,
                        inplace=True)
        for src, dst in patch.add :
            self._add_edges(src, dst)
        self.edges_count = self.g.edges_count
    def _add_edges (self, src, dst) :
        if not isinstance(src, int) :
            src = src.n
        if not isinstance(dst, int) :
            dst = dst.n
        if src == dst :
            return
        rules = self.g[src,dst]
        if not rules :
            return
        row = {"src" : src,
               "dst" : dst,
               "rule" : "|".join(sorted(rules))}
        if len(self.edges) :
            idx = self.edges.index.max() + 1
        else :
            idx = 0
        self.edges.loc[idx] = [row.get(c) for c in self.edges.columns]
    def merge (self, one, two, *rest) :
        """merge components
        Arguments:
         - one, two, ...: two or more component numbers
        """
        compo = [self.g[n] for n in (one, two) + rest]
        with log(head="merging",
                 tail="{step} (TIME: {time} | MEM: {memory:.1f}%)",
                 done_head="<b>merged:</b>",
                 done_tail=("%s => %s (TIME: {time})"
                            % (", ".join(str(c.n) for c in compo), compo[0].n)),
                 steps=["merging components", "updating tables"],
                 keep=True) :
            patch = self.g.merge(compo)
            log.update()
            # update nodes table
            drop = set(c.n for c in patch.components[1:])
            self.nodes.drop(index=self.nodes[self.nodes["node"].isin(drop)].index,
                            inplace=True)
            keep = patch.components[0]
            on, off = keep.on_off()
            succ = self.g.getsucc(keep.n)
            pred = self.g.getpred(keep.n)
            row = {"node" : keep.n,
                   "size" : len(keep),
                   "succ" : "|".join(str(s) for s in sorted(succ)),
                   "pred" : "|".join(str(s) for s in sorted(pred)),
                   "on" : on,
                   "off" : off,
                   "init" : keep.init,
                   "dead" : keep.dead,
                   "scc" : keep.scc,
                   "hull" : keep.hull}
            found = self.nodes[self.nodes["node"] == keep.n]
            if len(found) :
                idx = found.index[0]
            else :
                idx = self.nodes.index.max() + 1
            self.nodes.loc[idx] = [row.get(c) for c in self.nodes.columns]
            self.nodes_count = self.g.nodes_count
            # update edges table
            edges = self.edges[["src", "dst"]].apply(tuple, axis="columns")
            self.edges.drop(index=self.edges[edges.isin(patch.rem)].index,
                            inplace=True)
            for src, dst in patch.add :
                self._add_edges(src, dst)
            self.edges_count = self.g.edges_count
            log.update()
    def drop (self, first, *others) :
        """remove components
        Arguments:
         - first, ...: one or more components to remove
        """
        compo = [self.g[n] for n in (first,) + others]
        with log(head="dropping",
                 tail="{step} (TIME: {time} | MEM: {memory:.1f}%)",
                 done_head="<b>dropped:</b>",
                 done_tail=("%s (TIME: {time})" % ", ".join(str(c.n) for c in compo)),
                 steps=["#%s" % c.n for c in compo] + ["updating tables"],
                 keep=True) :
            for c in compo :
                self.g.del_compo(c.n)
                log.update()
            # update nodes table
            drop = set(c.n for c in compo)
            self.nodes.drop(index=self.nodes[self.nodes["node"].isin(drop)].index,
                            inplace=True)
            _dropn = re.compile("|".join("(\A|\D)%s(\D|\Z)" % n for n in drop))
            _dropb = re.compile("\\|{2,}")
            def _drop (s) :
                return _dropb.sub("|", _dropn.sub("", s)).strip("|")
            for col in ("succ", "pred") :
                self.nodes[col] = self.nodes[col].apply(_drop)
            self.nodes_count = self.g.nodes_count
            # update edges table
            self.edges.drop(index=self.edges[self.edges["src"].isin(drop)
                                             | self.edges["dst"].isin(drop)].index,
                            inplace=True)
            self.edges_count = self.g.edges_count
            log.update()
    def search (self, *states, col=None) :
        """search states through all the components and separate them
        Arguments:
         - states...: specification of searched states as in method split
        Options:
         - col (None): if not None, adds the result of search to a new column
        Return: a DataFrame summarising which states have been found in which components
        """
        with log(head="searching",
                 done_head="<b>found:</b>",
                 tail="{step} (TIME: {time} | MEM: {memory:.1f}%)",
                 steps=["%r" % s for s in states] + ["matching"]) :
            splitters = []
            for st in states :
                todo = [(c.n, c.s) for c in self]
                split = expr2sdd(st, self._name2sdd)
                splitters.append(split)
                for num, sdd in todo :
                    core = sdd & split
                    if core :
                        rest = sdd - core
                        succ = (self.g.succ_s & sdd)(core) & rest
                        rest -= succ
                        pred = (self.g.pred_s & sdd)(core) & rest
                        tosplit = num
                        for split in (s for s in (core, succ, pred) if s) :
                            patch = self.g.split_states(tosplit, split)
                            if patch :
                                self._patch_tables(patch)
                                tosplit = patch[-1].n
                    log.update()
            found = {}
            if col is None :
                def _has (row) :
                    c = self[row.node]
                    for i, s in enumerate(splitters) :
                        if c.s & s :
                            found.setdefault(i, set()).add(row.node)
                self.nodes.apply(_has, axis=1)
            else :
                def _has (row) :
                    h = []
                    c = self[row.node]
                    for i, s in enumerate(splitters) :
                        if c.s & s :
                            h.append(str(i))
                            found.setdefault(i, set()).add(row.node)
                    return "|".join(h)
                self.n[col] = _has
            for i, st in enumerate(states) :
                if not found.get(i) :
                    log.warn("%r not found" % st)
            return pd.DataFrame.from_records(((st, i, found.get(i, None))
                                              for i, st in enumerate(states)),
                                             columns=["query", "index", "matches"])
    def search_path (self, *states, col=None, prune=True) :
        """as search plus tries to find a path through the states
        Arguments:
         - states...: specification of searched states as in method split
        Options:
         - col (None): if not None, adds the result of search to a new column
         - prune (True): remove states and edges not involved in found path
        Return: a list of lists, each of which being a path from one searched state
           to another (except for the first one that is from the initial state to the
           first searched state)
        """
        found = self.search(*states, col=col)
        col = found.columns[-1]
        stops = ([set(c.n for c in self if c.init)]
                 + [row[col] for _, row in found.iterrows() if row[col]])
        if len(stops) < 2 :
            log.err("no path")
            return
        g = to_graph(self.nodes, "node", self.edges, "src", "dst", data=False)
        path = self._search_path(g, *stops)
        if prune :
            drop = set(c.n for c in self)
            for p in path :
                drop.difference_update(p)
            self.drop(*drop)
            keep = set()
            for p in path :
                keep.update(zip(p, p[1:]))
            self.edges["_"] = self.edges[["src", "dst"]].apply(tuple, axis=1)
            drop = self.edges[~self.edges["_"].isin(keep)].index
            self.edges.drop(index=drop, inplace=True)
            self.edges.drop(columns=["_"], inplace=True)
        text = ['<span style="color:#000088;">init</span>', "="]
        for s, p in zip(states, path) :
            text.append(" &gt; ".join("<code>%s</code>" % n for n in p))
            text.append("~")
            text.append('<span style="color:#000088;">%s</span>' % s)
            text.append("~")
        text.pop(-1)
        log.info(" ".join(text))
        return path
    def _search_path (self, g, one, two, *rest) :
        for src in one :
            for dst in two :
                try :
                    start = nx.shortest_path(g, src, dst)
                except nx.NetworkXNoPath :
                    continue
                if not rest :
                    return [start]
                suite = self._search_path(g, {dst}, *rest)
                if suite :
                    return [start] + suite
        return []

@help
class Model (_Model) :
    @cached_property
    def rr (self) :
        """show RR source code
        """
        h = HTML()
        with h("pre", style="line-height:140%") :
            sections = {}
            for d in self.spec.meta :
                sections.setdefault(d.kind, []).append(d)
            for sect, decl in sections.items() :
                with h("span", style="color:#008; font-weight:bold;", BREAK=True) :
                    h.write("%s:" % sect)
                for d in decl :
                    h.write("    ")
                    color = "#080" if d.state.sign else "#800"
                    with h("span", style="color:%s;" % color) :
                        h.write("%s" % d.state)
                    h.write(": %s\n" % d.description)
            for sect in ("constraints", "rules") :
                rules = getattr(self.spec, sect)
                with h("span", style="color:#008; font-weight:bold;", BREAK=True) :
                    h.write("%s:" % sect)
                for rule in rules :
                    h.write("    ")
                    for i, s in enumerate(rule.left) :
                        if i :
                            h.write(", ")
                        color = "#080" if s.sign else "#800"
                        with h("span", style="color:%s;" % color) :
                            h.write(str(s))
                    with h("span", style="color:#008; font-weight:bold;") :
                        h.write(" &gt;&gt; ")
                    for i, s in enumerate(rule.right) :
                        if i :
                            h.write(", ")
                        color = "#080" if s.sign else "#800"
                        with h("span", style="color:%s;" % color) :
                            h.write(str(s))
                    with h("span", style="color:#888;") :
                        h.write("   # %s" % rule.name())
                    h.write("\n")
        return h
    def charact (self, constraints: bool=True) -> pd.DataFrame :
        """compute variables static characterisation
        Options:
         - constraint (True): take constraints into account
        Return: a DataFrame which the counts characterising each variable
        """
        index = list(sorted(s.state.name for s in self.spec.meta))
        counts = pd.DataFrame(index=index,
                              data={"init" : np.full(len(index), 0),
                                    "left_0" : np.full(len(index), 0),
                                    "left_1" : np.full(len(index), 0),
                                    "right_0" : np.full(len(index), 0),
                                    "right_1" : np.full(len(index), 0)})
        for s in self.spec.meta :
            if s.state.sign :
                counts.loc[s.state.name,"init"] = 1
        if constraints :
            todo = [self.spec.constraints, self.spec.rules]
        else :
            todo = [self.spec.rules]
        for rule in itertools.chain(*todo) :
            for side in ["left", "right"] :
                for state in getattr(rule, side) :
                    sign = "_1" if state.sign else "_0"
                    counts.loc[state.name,side+sign] += 1
        return counts
    def draw_charact (self, constraints: bool=True, **options) :
        """draw a bar chart of nodes static characterization
        Options:
         - constraint (True): take constraints into account
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (300): figure height (in pixels)
         - fig_padding (0.01): internal figure margins
         - fig_title (None): figure title
        Bars options:
         - bar_spacing (1): additional space at the left of bars to fit variable names
         - bar_palette ("RGW"): color palette for bars
        """
        self.opt = opt = getopt(options,
                                fig_width=960,
                                fig_height=300,
                                fig_padding=0.01,
                                fig_title=None,
                                bar_spacing=1,
                                bar_palette="RGW")
        counts = self.charact(constraints)
        shift = (counts["left_0"] + counts["left_1"]).max() + opt.bar.spacing
        last = shift + (counts["right_0"] + counts["right_1"]).max()
        xs = bq.OrdinalScale(reverse=True)
        ys = bq.LinearScale(min=-opt.bar.spacing, max=last+1)
        bar_left = bq.Bars(x=counts.index,
                           y=[counts["left_0"], counts["left_1"]],
                           scales={"x": xs, "y": ys},
                           padding=0.1,
                           colors=Palette.mkpal(opt.bar.palette, 2),
                           orientation="horizontal")
        bar_right = bq.Bars(x=counts.index,
                            y=[counts["right_0"] + shift,
                               counts["right_1"] + shift],
                            scales={"x": xs, "y": ys},
                            padding=0.1,
                            colors=Palette.mkpal(opt.bar.palette, 2),
                            orientation="horizontal",
                            base=shift)
        ax_left = bq.Axis(scale=xs, orientation="vertical", grid_lines="none",
                          side="left", offset={"scale": ys, "value": 0})
        ax_right = bq.Axis(scale=xs, orientation="vertical", grid_lines="none",
                           side="left", offset={"scale": ys, "value": shift})
        ay = bq.Axis(scale=ys, orientation="horizontal",
                     tick_values=(list(range(shift + 1 - opt.bar.spacing))
                                  + list(range(shift, last+1))),
                     tick_style={"display": "none"})
        if opt.fig.title :
            fig = bq.Figure(marks=[bar_left, bar_right],
                            axes=[ax_left, ax_right, ay],
                            padding_x=opt.fig.padding, padding_y=opt.fig.padding,
                            layout=ipw.Layout(width="%spx" % opt.fig.width,
                                              height="%spx" % opt.fig.height),
                            fig_margin={"top" : 60, "bottom" : 0,
                                        "left" : 0, "right" : 0},
                            title=opt.fig.title)
        else :
            fig = bq.Figure(marks=[bar_left, bar_right],
                            axes=[ax_left, ax_right, ay],
                            padding_x=opt.fig.padding, padding_y=opt.fig.padding,
                            layout=ipw.Layout(width="%spx" % opt.fig.width,
                                              height="%spx" % opt.fig.height),
                            fig_margin={"top" : 0, "bottom" : 0,
                                        "left" : 0, "right" : 0})
        display(ipw.VBox([fig, bq.Toolbar(figure=fig)]))
    def __call__ (self, name:str, compact: bool=True, split: bool=True,
                  force: bool=False, profile: bool=False) -> ComponentView :
        """build a named view of the model
        Arguments:
         - name: directory where view's data is stored
        Options:
         - split (True): split SCC hull and related components
         - force (False): rebuild from scratch (instead of reloading from files)
         - compact (True): whether the view hides transient states and constraints or not
        """
        view = ComponentView(name, self, compact=compact)
        view.build(split=split, force=force, profile=profile)
        return view
    def compile (self, force=False, profile=False) :
        with log(head="<b>compiling</b>",
                 tail=self.path,
                 done_head="<b>loaded:</b>",
                 done_tail=self.path) :
            src_hash = sha512(open(self.path, "rb").read()).hexdigest()
            self.states = load(self.spec, self.path, self.base, src_hash, force, profile)
    def gal (self) :
        path = str(self["gal"])
        with log(head="<b>saving</b>",
                 tail=path,
                 done_head="<b>saved:</b>",
                 done_tail=path), open(path, "w") as out :
            dom = " && ".join("((%s == 0) || (%s == 1))" % (m.state.name, m.state.name)
                              for m in self.spec.meta)
            name = re.sub("[^a-z0-9]+", "", self.base.name, flags=re.I)
            out.write("gal %s {\n    //*** variables ***//\n" % name)
            for sort in self.spec.meta :
                out.write("    // %s: %s (%s)\n"
                          % (sort.state, sort.description, sort.kind))
                out.write("    int %s = %s;\n" % (sort.state.name, int(sort.state.sign)))
            out.write("    //*** constraints ***//\n")
            guards = []
            for const in self.spec.constraints :
                guard = " && ".join("(%s == %s)" % (s.name, int(s.sign))
                                    for s in const.left)
                loop = " && ".join("(%s == %s)" % (s.name, int(s.sign))
                                   for s in const.right)
                out.write("    // %s\n" % const)
                out.write("    transition C%s [%s && (!(%s)) && %s] {\n"
                          % (const.num, guard, loop, dom))
                for s in const.right :
                    out.write("        %s = %s;\n" % (s.name, int(s.sign)))
                out.write("    }\n")
                guards.append("%s && (!(%s))" % (guard, loop))
            if guards :
                prio = "(!(%s))" % " || ".join("(%s)" % g for g in guards)
            else :
                prio = "true"
            out.write("    //*** rules ***//\n")
            for rule in self.spec.rules :
                guard = " && ".join("(%s == %s)" % (s.name, int(s.sign))
                                    for s in rule.left)
                loop = " && ".join("(%s == %s)" % (s.name, int(s.sign))
                                   for s in rule.right)
                out.write("    // %s\n" % rule)
                out.write("    transition R%s [%s && (!(%s)) && %s && %s] {\n"
                          % (rule.num, guard, loop, prio, dom))
                for s in rule.right :
                    out.write("        %s = %s;\n" % (s.name, int(s.sign)))
                out.write("    }\n")
            out.write("}\n")
    ##
    ## ecosystemic (hyper)graph
    ##
    def ecograph (self, constraints: bool=True, **opt) :
        """draw the ecosystemic graph
        Options:
         - constraint (True): take constraints into account
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (600): figure height (in pixels)
         - fig_padding (0.01): internal figure margins
         - fig_title (None): figure title
        Graph options:
         - graph_layout ("circo"): layout engine to compute nodes positions
        Nodes options:
         - nodes_label ("node"): column that defines nodes labels
         - nodes_shape (auto): column that defines nodes shape
         - nodes_size (35): nodes width
         - nodes_color ("init"): column that defines nodes colors
         - nodes_selected ("#EE0000"): color of the nodes selected for inspection
         - nodes_palette ("RGW"): palette for the nodes colors
         - nodes_ratio (1.2): height/width ratio of non-symmetrical nodes
        """
        nodes = []
        for state in sorted(s.state for s in self.spec.meta) :
            nodes.append((state.name, state.sign))
        edges = []
        if constraints :
            todo = [self.spec.constraints, self.spec.rules]
        else :
            todo = [self.spec.rules]
        for rule in itertools.chain(*todo) :
            for src in rule.left :
                for dst in rule.right :
                    edges.append((src.name, dst.name, rule.name(), rule.text()))
        return Graph(pd.DataFrame.from_records(nodes, columns=["node", "init"]),
                     pd.DataFrame.from_records(edges,
                                               columns=["src", "dst", "rule", "rr"]),
                     defaults={
                         "graph_layout" : "circo",
                         "nodes_color" : "init",
                         "nodes_colorscale" : "discrete",
                         "nodes_palette" : "RGW",
                         "nodes_shape" : "circ",
                         "marks_shape" : None
                     }, **opt)
    def ecohyper (self, constraints: bool=True, **opt) :
        """draw the ecosystemic hypergraph
        Options:
         - constraint (True): take constraints into account
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (600): figure height (in pixels)
         - fig_padding (0.01): internal figure margins
         - fig_title (None): figure title
        Graph options:
         - graph_layout ("fdp"): layout engine to compute nodes positions
        Nodes options:
         - nodes_size (35): nodes width
         - nodes_selected ("#EE0000"): color of the nodes selected for inspection
        """
        nodes = []
        edges = []
        if constraints :
            rules = list(itertools.chain(self.spec.constraints, self.spec.rules))
        else :
            rules = list(self.spec.rules)
        for state, desc in sorted((s.state, s.description) for s in self.spec.meta) :
            nodes.append((state.name, desc, "circ", 0 if state.sign else 1))
        for rule in rules :
            name = rule.name()
            nodes.append((name, rule.text(), "sbox", 2 if name[0] == "C" else 3))
            for var in set(s.name for s in itertools.chain(rule.left, rule.right)) :
                v0 = State(var, False)
                v1 = State(var, True)
                if v1 in rule.left :
                    if v1 in rule.right :
                        edges.append((var, name, "*", "*"))
                    elif v0 in rule.right :
                        edges.append((var, name, "o", "*"))
                    else :
                        edges.append((var, name, "-", "*"))
                elif v0 in rule.left :
                    if v1 in rule.right :
                        edges.append((var, name, "*", "o"))
                    elif v0 in rule.right :
                        edges.append((var, name, "o", "o"))
                    else :
                        edges.append((var, name, "-", "o"))
                elif v1 in rule.right :
                    edges.append((var, name, "*", "-"))
                elif v0 in rule.right :
                    edges.append((var, name, "o", "-"))
        return Graph(pd.DataFrame.from_records(nodes,
                                               columns=["node", "info", "shape",
                                                        "color"]),
                     pd.DataFrame.from_records(edges,
                                               columns=["src", "dst", "get", "set"]),
                     defaults={
                         "graph_layout" : "fdp",
                         "graph_directed" : False,
                         "nodes_shape" : "shape",
                         "nodes_color" : "color",
                         "nodes_colorscale" : "discrete",
                         "nodes_palette" : ["#AAFFAA", "#FFAAAA", "#FFFF88", "#FFFFFF"],
                         "edges_tips" : ["get", "set"],
                         "gui_main" : [["layout", "size"],
                                       "figure",
                                       "toolbar",
                                       "inspect"],
                     }, **opt)
    ##
    ## Petri nets
    ##
    def petri (self,ra=True) :
        n = ptnet.net.Net()
        places = {}
        for state in (s.state for s in self.spec.meta) :
            places[state] = n.place_add(str(state), 1)
            places[~state] = n.place_add(str(~state))
        for rule in self.spec.rules :
            for i, r in enumerate(rule.normalise()) :
                if set(r.left) == set(r.right) :
                    continue
                t = n.trans_add("%s.%s" % (rule.name(), i))
                for v in r.vars() :
                    on = State(v, True)
                    off = State(v, False)
                    if on in r.left and on in r.right :
                        if ra == True:
                            t.cont_add(places[on])
                        else:
                            t.pre_add(places[on])
                            t.post_add(places[on])
                    elif on in r.left and off in r.right :
                        t.pre_add(places[on])
                        t.post_add(places[off])
                    elif off in r.left and off in r.right :
                        if ra == True:
                            t.cont_add(places[off])
                        else:
                            t.pre_add(places[off])
                            t.post_add(places[off])
                    elif off in r.left and on in r.right :
                        t.pre_add(places[off])
                        t.post_add(places[on])
        return n
    def unfold (self,unf="cunf",rule="mcm",ra=True) :
        n = None
        if unf == "punf" :
            n = self.petri(ra=False)
        elif unf == "cunf" :
            n = self.petri(ra)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".pnml") as pep,\
             tempfile.NamedTemporaryFile("rb", suffix=".pnml") as cuf:
            if unf=="punf":
                n.write(pep,fmt='pnml')

                n.write(sys.stdout,fmt='pnml')
            else:
                n.write(pep)
            pep.flush()
            if unf == "cunf":
                os.system("cunf -c %s -s %s %s" % (rule,cuf.name, pep.name))
            elif unf == "punf":
                os.system("punf -f=%s -m=%s" % (pep.name, cuf.name))
            u = ptnet.unfolding.Unfolding()
            if unf=="punf":
                u.read(cuf,fmt='pnml')
            else:
                u.read(cuf)
        return u

__extra__ = ["Model", "ComponentView", "ExplicitView", "parse", "Palette"]
