# -*- coding: utf-8 -*-
import threading
import time
import inspect
import ctypes

class base_algorithm_class(object):
    def __init__(self,inp,res):
        self.result_cache = res
        self.input_cache = inp
        pass
    def algorithm_interface(self,input_corpus):
        pass
    def run_work(self):
        pass

__auto_start__ = 1      
__prohibit_start__ = 0  


class base_thread_class(threading.Thread):
    def __init__(self,input_class,auto_flag=__prohibit_start__):
        threading.Thread.__init__(self)
        if isinstance(input_class,base_algorithm_class) or isinstance(input_class,algorithm_manager_class):
            self.run_class = input_class
        else:
            raise RuntimeError('Passing the algorithm type is incorrect, or maybe they are not common basic class.')
        if auto_flag == __auto_start__:
            self.start()
        pass
    def run(self):
        while True:
            self.run_class.run_work()
            time.sleep(0.01)
        pass
    def __async_raise(self,tid, exctype):
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
            if res == 0:
                raise ValueError("invalid thread id")
            elif res != 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
                raise SystemError("PyThreadState_SetAsyncExc failed")
      
    def stop_thread(self,thread):
        self.__async_raise(thread.ident, SystemExit)

class base_cache_class(object):
    def __init__(self,mutex):
        self.cache = []
        if type(mutex) == type(threading.Lock()):
            self.mutex = mutex
        else:
            raise RuntimeError("passing the lock parameter is incorrect.")  
        pass
    def push_cache(self,element):
        if self.mutex.acquire(1):
            self.cache.append(element)
            self.mutex.release()
        pass
    def get_cache(self):      
        if self.mutex.acquire(1):
            if len(self.cache) > 0:
                elem = self.cache.pop()
            self.mutex.release()
            return elem
        pass
    def get_index_cache(self,index):
        if self.mutex.acquire(1):
             if len(self.cache) > 0:
                 elem = self.cache[index]
             self.mutex.release()
             return elem
    def get_cache_len(self):
        if self.mutex.acquire(1):
            cache_len = len(self.cache)
            self.mutex.release()
            return cache_len
    def get_whole_cache(self):
        if self.mutex.acquire(1):
            cache = []
            if len(self.cache) > 0:
                cache = self.cache
                self.cache = []
            self.mutex.release()
            return cache
    
    def input_batch_cache(self,batch_cache=[]):
        if self.mutex.acquire(1):
            self.cache.extend(batch_cache)
            self.mutex.release()
        pass

    def __getitem__(self,index):
        return self.cache[index]
    
    def delete_item_cache(self,obj):
        if self.mutex.acquire(1):
            self.cache.remove(obj)
            self.mutex.release()
        pass

    def insert_item_cache(self,index,obj):
        if self.mutex.acquire(1):
            self.cache.insert(index,obj)
            self.mutex.release()
        return obj
        pass

class thransfer_cache(base_cache_class):
    def __init__(self,mutex):
        base_cache_class.__init__(self,mutex)
        pass

    def thransfer_from(self,from_cache):
        self.input_batch_cache(from_cache.get_whole_cache())
        pass

class attach_cache_thread_class(base_thread_class):
    def __init__(self,input_algorithm_,algorithm_name = 'None'):
        base_thread_class.__init__(self,input_algorithm_,auto_flag=__prohibit_start__)
        self.algorithm_name = algorithm_name
        """
        if isinstance(input_algorithm_,base_algorithm_class):
            input_algorithm_.input_cache = input_cache
            input_algorithm_.result_cache = result_cache
        else:
            raise RuntimeError('passing the algorithm type is correct, or maybe they are not common basic class.')
        pass
        """
class specific_algorithm_class(base_algorithm_class):
    def __init__(self,
                input_algorithm,
                input_c,
                result_c,
                resource_index = 0,
                ):
        base_algorithm_class.__init__(self,input_c,result_c)
        self.algorithm_model = input_algorithm 
        self.resource_index = resource_index
        pass
    def run_work(self):
        #fix  code
        if self.input_cache.get_cache_len() > 0:
            input_ = self.input_cache.get_whole_cache()
            result_predict = self.algorithm_model.prediction(input_,self.resource_index)
            re = thransfer_cache(threading.Lock())
            re.input_batch_cache(result_predict)
            self.result_cache.thransfer_from(re)
        pass

class temporary_test_algorithm:
    def prediction(self,obj,filter_tag=-1):
        return_str = []
        for str in obj:
            return_str.append(str+' + TWO_CLASSIFY')
        return return_str
class second_test_algorithm:
    def prediction(self,obj,filter_tag=-1):
        return_str = []
        for str in obj:
            return_str.append(str+' + ENTITY_FIND')
        return return_str


class algorithm_manager_class(base_thread_class):
    def __init__(self):
        base_thread_class.__init__(self,self,__prohibit_start__)
        self.__algorithm_list = base_cache_class(threading.Lock()) #algorithm_list
        pass
    def __append_obj(self,obj):
        self.__algorithm_list.push_cache(obj)
        pass
    def __get_index_obj(self,index):
        return self.__algorithm_list.get_index_cache(index)

    def __launch_whole_algorithm(self):
        len = self.__algorithm_list.get_cache_len()
        for index in range(len):
            item = self.__get_index_obj(index)
            if isinstance(item,base_thread_class):
                if 0 == __prohibit_start__:
                    item.start()
                    pass
        self.start()
        pass

    def delete_algorithm_item(self,algorithm_type='None',force = True):
        for index in range(self.__algorithm_list.get_cache_len()):
            if self.__algorithm_list[index].algorithm_name == algorithm_type:
                if force:
                    self.__algorithm_list[index].stop_thread(self.__algorithm_list[index])
                else:
                    while True and force == False: 
                        in_len = self.__algorithm_list[index].run_class.input_cache.get_cache_len() 
                        out_len = self.__algorithm_list[index].run_class.result_cache.get_cache_len() 
                        if in_len == 0 and out_len == 0:
                            self.__algorithm_list[index].stop_thread(self.__algorithm_list[index])
                            break
                temporary_cache = self.__algorithm_list[index].run_class.input_cache.get_whole_cache()
                temporary_cache = []
                temporary_cache = self.__algorithm_list[index].run_class.result_cache.get_whole_cache()
                temporary_cache = []
                self.__algorithm_list.delete_item_cache(self.__algorithm_list[index])
                return algorithm_type
        return 'None'

    def insert_algorithm_item(self,index,algorithm_type,filter_iter):
        run_new_item = self.__algorithm_list.insert_item_cache(index,self.__create_fectory(algorithm_type,filter_iter))
        run_new_item.start()
        pass

    def detect_algorithm_list(self):
        return [[x.algorithm_name,x.run_class.resource_index] for x in self.__algorithm_list]

    def run_work(self):
        #print('join with the manage thread %d!'%self.__algorithm_list.get_cache_len())
        for index in range(self.__algorithm_list.get_cache_len() - 1):
            self.__algorithm_list[index+1].run_class.input_cache.thransfer_from(
                    self.__algorithm_list[index].run_class.result_cache)
    
    def input_corpus_predict(self,corpus):
        if self.__algorithm_list.get_cache_len() > 0:
            self.__algorithm_list[0].run_class.input_cache.input_batch_cache(corpus)
        else:
            raise RuntimeError('error: algorithm list is empty.')
        
    def __create_fectory(self,model_type,resource_index):
        if model_type == 'TWO_CLASSIFY':
            return attach_cache_thread_class(
                    input_algorithm_ = specific_algorithm_class(input_algorithm=temporary_test_algorithm(),
                input_c=thransfer_cache(mutex=threading.Lock()),
                result_c=thransfer_cache(mutex=threading.Lock()),resource_index=resource_index),algorithm_name=model_type)
            pass
        if model_type == 'ENTITY_FIND':
            return attach_cache_thread_class(
                    input_algorithm_ = specific_algorithm_class(input_algorithm=second_test_algorithm(),
                input_c=thransfer_cache(mutex=threading.Lock()),
                result_c=thransfer_cache(mutex=threading.Lock()),resource_index=resource_index),algorithm_name=model_type)
            pass

    def run_manager(self,type_list = [], filter_index = []):
        try:
            type_len = len(type_list)
            filter_len = len(filter_index)
            assert type_len == filter_len
            for (type_iter,index_iter) in zip(type_list,filter_index):
                self.__append_obj(self.__create_fectory(type_iter,index_iter))
            self.__launch_whole_algorithm()
            return 1
        except:
            return 0
        finally:
            pass
        pass

    def result_corpus_predict(self):
        if self.__algorithm_list.get_cache_len() > 0:
            return self.__algorithm_list[self.__algorithm_list.get_cache_len()-1].run_class.result_cache.get_whole_cache()
            pass
        else:
            return []

if __name__ == '__main__':
    specify_algorithm_manager = algorithm_manager_class()
    specify_algorithm_manager.run_manager(['TWO_CLASSIFY','ENTITY_FIND','TWO_CLASSIFY'],[0,0,0])
    specify_algorithm_manager.input_corpus_predict(['to what can our life  on the earth could be likened','I love wiseweb'])
    print(specify_algorithm_manager.detect_algorithm_list())
    time.sleep(1)
    print(specify_algorithm_manager.result_corpus_predict())
    time.sleep(1) 
    print(specify_algorithm_manager.insert_algorithm_item(0,'ENTITY_FIND',0))
    time.sleep(1)
    print(specify_algorithm_manager.detect_algorithm_list())
    specify_algorithm_manager.input_corpus_predict(['to what can our life  on the earth could be likened','I love webside'])
    time.sleep(1)
    print(specify_algorithm_manager.result_corpus_predict())
    
    #specify_algorithm_manager.input_corpus_predict(['to what our life  on the earth could be likened','I love webside'])
    print(specify_algorithm_manager.delete_algorithm_item('TWO_CLASSIFY'))
    specify_algorithm_manager.input_corpus_predict(['to what can our life  on the earth could be likened','I love wiseweb'])
    print(specify_algorithm_manager.detect_algorithm_list())
    time.sleep(1)
    print(specify_algorithm_manager.result_corpus_predict())
    
    specify_algorithm_manager.input_corpus_predict(['to what can our life  on the earth could be likened','I love webside'])
    time.sleep(1)
    print(specify_algorithm_manager.result_corpus_predict())

