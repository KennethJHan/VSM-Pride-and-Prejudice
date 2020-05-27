import sys
import pandas as pd

class chapter_reader:
    @staticmethod
    def read_chap(num: int) -> str:
        chap_str = ""
        with open(f"../data/chap_{num}.txt", 'r', encoding="utf8") as fr:
            print(f"Read chapter {num} ...")
            l = fr.readlines()
            assert(len(l) == 1)
            chap_str = l[0].strip()
            return chap_str

class vsm_tool:
    @staticmethod
    def token_string(string: str) -> str:
        from tensorflow.keras.preprocessing.text import text_to_word_sequence
        token_list = text_to_word_sequence(string)
        token_except_stop_list = []
        import nltk
        #nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english')) 
        for token in token_list:
            if token not in stop_words:
                token_except_stop_list.append(token)
                
        #return " ".join(token_list)
        return " ".join(token_except_stop_list)
    
    @staticmethod
    def make_bow(doc_list: list) -> dict:
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english')) 
        bow = dict()
        for doc in doc_list:
            string = doc.split()
            for word in string:
                if word in stop_words:
                    continue
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
        return bow

    @classmethod
    def tf(self, t: str, d: str) -> int:
        """
            return occurence number of term from document
        """
        return d.count(t)
    
    @classmethod
    def idf(self, t: str, doc_list: list) -> float:
        from math import log
        df = 0
        for doc in doc_list:
            df += t in doc
        return log(len(doc_list)/(df + 1))

    @classmethod
    def tf_idf(self, t: str, d: str, doc_list: list) -> float:
        return self.tf(t, d) * self.idf(t, doc_list)

    @staticmethod
    def calc_cosine_similarity(doc1, doc2):
        from numpy import dot
        from numpy.linalg import norm
        import numpy as np
        return dot(doc1, doc2)/(norm(doc1)*norm(doc2))

    @staticmethod
    def print_result(q: str, l: list):
        num = 0
        max_val = l[0][0]
        for i in range(len(l)):
            #print(i, num, max_val, l[i][0], max_val < l[i][0])
            if max_val < l[i][0]:
                max_val = l[i][0]
                num = i
        #num = 2*(num+1)-1
        #print(f"#Query: {q}")
        #print(f"#Result: Chap_{num}\n----------")
        #for i in range(len(l)):
        #    print(f"Chap_{2*(i+1)-1}: {l[i][0]}")
        ll = []
        for i in l:
            ll.append(float(i[0]))
        freq_result = "\t".join(list(map(str,ll)))
        #print(f"{num+1}\t{freq_result}")
        print(f"{num+1}\t{freq_result}\t{q}")

class VectorSpaceModelRunner:
    def __init__(self, query_file: str):
        #self.query_file = query_file
        #self.query_raw_string = self.read_query(query_file) # query가 하나의 문장인 경우 사용
        self.queries_list = self.read_query_to_list(query_file) # 한 텍스트에 여러 쿼리가 들어있는 경우 사용
        #self.to_read_chap_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        #self.to_read_chap_list = list(range(1,11,1))
        self.to_read_chap_list = list(range(1,61,1))
        self.bow, self.doc_list, self.tf_idf_result = self.read_chapter()

    def read_query(self, query_file: str) -> str:
        print("read_query")
        query_raw_string = ""
        with open(query_file, 'r', encoding="utf-8") as fr:
            for line in fr:
                query_raw_string += line.strip()
        return query_raw_string

    def read_query_to_list(self, query_file: str) -> list:
        print("read_query_to_list")
        queries_list = []
        with open(query_file,'r',encoding="utf-8") as fr:
            for line in fr:
                queries_list.append(line.strip())
        return queries_list

    def process_query(self):
        print("process_query")
        # query가 하나의 문장인 경우 사용
        #token_query = vsm_tool.token_string(self.query_raw_string) 
        # #bow_query = vsm_tool.make_bow([token_query])
        # result_query = []
        # k = list(self.bow.keys())
        # for t in k:
        #     result_query.append(vsm_tool.tf_idf(t, token_query, self.doc_list))

        # z = {'bow':k, 'rq':result_query}
        # d = pd.DataFrame.from_dict(z).T
        # d.columns = d.iloc[0]
        # d = d.drop(d.index[0])
        # from sklearn.metrics.pairwise import cosine_similarity
        # #print(d)
        # similarities_query = cosine_similarity(self.tf_idf_result, d)
        # vsm_tool.print_result(self.query_raw_string, similarities_query)


        # 한 텍스트에 여러 쿼리가 들어있는 경우 사용
        #print(self.queries_list)
        for query_raw_string in self.queries_list:
            #print(query_raw_string)
            token_query = vsm_tool.token_string(query_raw_string)

            #bow_query = vsm_tool.make_bow([token_query])
            result_query = []
            k = list(self.bow.keys())
            for t in k:
                result_query.append(vsm_tool.tf_idf(t, token_query, self.doc_list))
            z = {'bow':k, 'rq':result_query}
            d = pd.DataFrame.from_dict(z).T
            d.columns = d.iloc[0]
            d = d.drop(d.index[0])
            from sklearn.metrics.pairwise import cosine_similarity
            #print(d)
            similarities_query = cosine_similarity(self.tf_idf_result, d)
            #print(similarities_query)
            # [[0.42911701]
            # [0.11394441]
            # [0.08863185]
            # [0.14744876]
            # [0.1408225 ]
            # [0.10029122]
            # [0.11637492]
            # [0.12766614]
            # [0.13003495]
            # [0.14278472]]
            #vsm_tool.print_result(query_raw_string, similarities_query)
            vsm_tool.print_result(token_query, similarities_query)

    
    def read_chapter(self) -> tuple:
        print("read_chapter")
        doc_list = []
        for chap_num in self.to_read_chap_list:
            chap_string = chapter_reader.read_chap(chap_num)
            token_chap_string = vsm_tool.token_string(chap_string)
            doc_list.append(token_chap_string)
        bow = vsm_tool.make_bow(doc_list)

        result = []
        for d in doc_list:
            result.append([])
            for t in bow:
                result[-1].append(vsm_tool.tf_idf(t, d, doc_list))

        tf_idf_result = pd.DataFrame(result, columns = bow)

        return bow, doc_list, tf_idf_result




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"#usage: python {sys.argv[0]} [query.txt]")
        sys.exit()

    query_file = sys.argv[1]
    vsm_runner = VectorSpaceModelRunner(query_file)
    #print(vsm_runner.bow)
    vsm_runner.process_query()
