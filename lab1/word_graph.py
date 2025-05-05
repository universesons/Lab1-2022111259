import re
from collections import defaultdict, Counter
import random
import heapq
import os
from typing import List, Tuple, Dict, Set, Optional


#文本输入处理类
class TextProcessor:
    """处理文本输入，提取单词序列"""
    @staticmethod
    def process_file(filename: str) -> List[str]:
        """从文件读取并处理文本"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            return TextProcessor.process_text(text)
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

    @staticmethod
    def process_text(text: str) -> List[str]:
        """处理文本字符串"""
        text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
        return [word for word in text.split() if word]

class WordGraph:
    """实现所有图操作的核心类"""
    def __init__(self):
        self.nodes: Set[str] = set()
        self.adj: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pagerank: Dict[str, float] = {}

    def build_graph(self, words: List[str]) -> None:
        """从单词序列构建图"""
        for i in range(len(words) - 1):
            from_word, to_word = words[i], words[i+1]
            self.nodes.add(from_word)
            self.nodes.add(to_word)
            self.adj[from_word][to_word] += 1

    def show_directed_graph(self, output_img: bool = False) -> None:
        """展示有向图结构"""
        print("\nDirected Graph (Adjacency List):")
        for from_word in sorted(self.nodes):
            edges = self.adj[from_word]
            if edges:
                print(f"{from_word} -> {', '.join(f'{to}({w})' for to, w in edges.items())}")
            else:
                print(f"{from_word} -> (no edges)")
        
        if output_img:
            self._generate_graph_image()

    def query_bridge_words(self, word1: str, word2: str) -> List[str]:
        """查询桥接词"""
        word1, word2 = word1.lower(), word2.lower()
        if word1 not in self.nodes or word2 not in self.nodes:
            return []
        return [w for w in self.nodes 
                if self.adj[word1].get(w, 0) > 0 and self.adj[w].get(word2, 0) > 0]

    def generate_new_text(self, text: str) -> str:
        """生成插入桥接词的新文本"""
        words = TextProcessor.process_text(text)
        if not words:
            return ""
            
        result = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            result.append(w1)
            bridges = self.query_bridge_words(w1, w2)
            if bridges:
                result.append(random.choice(bridges))
        
        result.append(words[-1])
        return ' '.join(result)

    def calc_shortest_path(self, word1: str, word2: Optional[str]) -> Tuple[List[Tuple[str, str]], int]:
        """计算最短路径"""
        word1 = word1.lower()
        if word1 not in self.nodes:
            return [], 0

        # Dijkstra算法实现
        distances = {node: float('inf') for node in self.nodes}
        distances[word1] = 0
        predecessors = {node: None for node in self.nodes}
        heap = [(0, word1)]
        
        while heap:
            dist, current = heapq.heappop(heap)
            if word2 is not None and current == word2.lower():
                break
                
            for neighbor, weight in self.adj[current].items():
                if dist + weight < distances[neighbor]:
                    distances[neighbor] = dist + weight
                    predecessors[neighbor] = current
                    heapq.heappush(heap, (dist + weight, neighbor))

        if word2 is None:
            # 输出从 word1 到所有节点的最短路径和距离
            paths = []
            max_distance = 0
            print(f"\nShortest paths from {word1}:")
            for node in sorted(self.nodes):
                if node == word1 or distances[node] == float('inf'):
                    continue
                path = []
                current = node
                while predecessors[current] is not None:
                    path.append((predecessors[current], current))
                    current = predecessors[current]
                path.reverse()
                paths.append(path)
                max_distance = max(max_distance, distances[node])
                path_str = ' -> '.join(f'{u}->{v}' for u, v in path)
                print(f"To {node}: {path_str}, Length: {distances[node]}")
            return paths, max_distance
        
        # 处理特定目标节点的情况
        target = word2.lower() if word2 else None
        if target not in self.nodes:
            return [], 0
            
        if distances[target] == float('inf'):
            return [], 0
            
        # 重构路径
        path = []
        current = target
        while predecessors[current] is not None:
            path.append((predecessors[current], current))
            current = predecessors[current]
        path.reverse()
        
        return path, distances[target]

            

    def calc_pagerank(self, word_freq: Dict[str, int], d: float = 0.85, iterations: int = 10) -> Dict[str, float]:
        """计算PageRank值"""
        N = len(self.nodes)
        if N == 0:
            return {}
        
        pr = {}
        total_word = sum(word_freq.values())
        for node in self.nodes:
            # TF: 词频 / 总词数
            tf = word_freq.get(node, 0) / total_word if total_word > 0 else 0
            # IDF: 1 / (出度 + 入度 + 1)
            out_degree = sum(self.adj[node].values())
            in_degree = sum(self.adj[src][node] for src in self.nodes)
            idf = 1 / (out_degree + in_degree + 1)
            pr[node] = tf * idf
        
        pr_sum = sum(pr.values())
        if pr_sum > 0:
            pr = {node: value / pr_sum for node, value in pr.items()}
        else:
        # 初始化
            pr = {node: 1/N for node in self.nodes}
        out_degree = {node: sum(self.adj[node].values()) for node in self.nodes}
        
        # 迭代计算
        for _ in range(iterations):
            new_pr = {}
            for node in self.nodes:
                incoming = sum(pr[src] * self.adj[src][node] / out_degree[src] 
                             for src in self.nodes if out_degree[src] > 0)
                new_pr[node] = (1 - d)/N + d * incoming
            pr = new_pr
        
        self.pagerank = pr
        return pr

    def random_walk(self) -> str:
        """随机游走"""
        if not self.nodes:
            return "Graph is empty"
            
        current = random.choice(list(self.nodes))
        path = [current]
        visited_edges = set()
        
        while True:
            neighbors = list(self.adj[current].keys())
            if not neighbors:
                break
                
            next_node = random.choice(neighbors)
            edge = (current, next_node)
            
            if edge in visited_edges:
                path.append(next_node)
                break
                
            visited_edges.add(edge)
            path.append(next_node)
            current = next_node
            print(current, end=' ->\n')
            # 用户可随时停止
            user_input = input("Press 'q' to stop, any other key to continue: ").strip()
            if user_input.lower() == 'q':
                break
        
        result = ' '.join(path)
        with open('random_walk.txt', 'w') as f:
            f.write(result)
        return result

    def _generate_graph_image(self) -> None:
        """生成图形可视化(需要安装graphviz)"""
        try:
            from graphviz import Digraph
            dot = Digraph(comment='Word Graph',format='png')
            
            for node in self.nodes:
                dot.node(node)
                
            for from_node in self.adj:
                for to_node, weight in self.adj[from_node].items():
                    dot.edge(from_node, to_node, label=str(weight))
            
            dot.render('word_graph.gv', view=False, cleanup=False)
            print("Graph image generated as word_graph.png")
        except ImportError:
            print("Graphviz not installed, skipping image generation")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python word_graph.py <filename>")
        return
    
    filename = sys.argv[1]
    words = TextProcessor.process_file(filename)
    if not words:
        return
        
    graph = WordGraph()
    graph.build_graph(words)

    word_freq = Counter(words)
    
    while True:
        print("\nMenu:")
        print("1. Show directed graph")
        print("2. Query bridge words")
        print("3. Generate new text with bridge words")
        print("4. Calculate shortest path")
        print("5. Calculate PageRank")
        print("6. Random walk")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            graph.show_directed_graph(output_img=True)
            
        elif choice == '2':
            word1 = input("Enter first word: ").strip()
            word2 = input("Enter second word: ").strip()
            bridges = graph.query_bridge_words(word1, word2)
            if not bridges:
                if word1 not in graph.nodes or word2 not in graph.nodes:
                    print(f"No '{word1}' or '{word2}' in the graph!")
                else:
                    print(f"No bridge words from {word1} to {word2}!")
            else:
                print(f"The bridge words from {word1} to {word2} are: {', '.join(bridges)}")
                
        elif choice == '3':
            text = input("Enter a new text: ").strip()
            new_text = graph.generate_new_text(text)
            print("Generated text:", new_text)
            
        elif choice == '4':
            word1 = input("Enter start word: ").strip()
            word2 = input("Enter end word: ").strip()
            if word1 not in graph.nodes or word2 not in graph.nodes:
                print(f"No '{word1}' or '{word2}' in the graph!")
                continue
            path, length = graph.calc_shortest_path(word1, word2)
            
            if word2:
                path, length = graph.calc_shortest_path(word1, word2)
                if path:
                    # 格式化路径：仅显示节点序列
                    path_nodes = [path[0][0]] + [edge[1] for edge in path]
                    path_str = ' -> '.join(path_nodes)
                    print(f"Shortest path from {word1} to {word2}: {path_str}")
                    print(f"Path length: {length}")
            else:
                paths, max_distance = graph.calc_shortest_path(word1, None)
                if paths:
                    print(f"Maximum distance: {max_distance}")
                
        elif choice == '5':
            pr = graph.calc_pagerank(word_freq)
            print("\nTop 10 PageRank values:")
            for node, score in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"{node}: {score:.4f}")
                
        elif choice == '6':
            print("Starting random walk...")
            path = graph.random_walk()
            print("Random walk path:", path)
            print("Result saved to random_walk.txt")
            
        elif choice == '7':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()