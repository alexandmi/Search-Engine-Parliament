from operator import imod
import numpy as np
from sqlalchemy import true 
from query_processor import query_search

if __name__=="__main__":
    
    while(true):

        # Loop for correct query input
        while(true):
            query = input("\nInsert your query: ")
            if query.strip():
                break
            print("\nPlease do not insert an empty query")

        # Loop for correct top-k results input
        while(true):
            num = input("\nInsert the number of top relevant results you want: ")
            if not(num.isnumeric()):
                print("\nPlease give a valid integer number.")
                continue
            num = int(num)
            if(num >= 1):
                break
            print("\nPlease give a valid integer number.")

        query_data, full_query_vector = query_search(query)
    
        print("\n")
        print(query_data[0:num])