#importing sentence transformer , cosine similarity from torch ilbrary

from sentence_transformers import SentenceTransformer, util
import torch
from torch.nn.functional import cosine_similarity

from torch.nn.functional import cosine_similarity
#similar_prod_indexes = []

#installing sentence transformer library

!pip install sentence_transformers

def similar_products(model_name,data1,data2,path_to_export):


  '''take model, two dataframe and path to export as input,
  return final database based on similarity,
  and export data to a location'''


  def embedded_model(model_name,data1,data2):
    '''create model of sentence transformer, take input as two dataframe and return cosine similarity'''
    # Creating a model of sentence transformer class
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Computing embedding for both amazon and flipkart product 
    df1_embedding = model.encode(data1.product_name,convert_to_tensor=True)                      #.to_numpy()
    df2_embedding = model.encode(data2.product_name,convert_to_tensor=True)

    # Computing cosine similarities
    cosine_sim_scores = util.cos_sim(df1_embedding,df2_embedding)

    return cosine_sim_scores


  def store_similar_prod_index(cosine_sim_scores,df1,df2):
    '''input tensor of cosine scores, two data frame and store index of similar products'''
    
    length = len(df1)
    
    if len(df1) > len(df2):
      length =len(df2)
    else:
      length = len(df1)

    # Storing the index of a similar Flipkart product corresponding to each Amazon product 
    similar_prod_indexes = []
    
    for i in range(length):
      max_score_idxs = torch.topk(cosine_sim_scores[i],k=5,largest=True,sorted=True).indices
      for idx in max_score_idxs:
        if idx != i:
          similar_prod_indexes.append(idx)
          break
    return similar_prod_indexes
  
  
  def create_sim_list(similar_prod_indexes):
    
    '''create list of similar product index'''

    #similar_prod_indexes = [x.item() for x in similar_prod_indexes]
    sim_index = [x.item() for x in similar_prod_indexes]
    
    return sim_index

  
  def extract_similar_data(similar_prod_indexes,data1,data2):

    '''input two dataframe and list of similar product index and extract details from dataframe'''
    
    # Extracting the details such as retail price and discounted price for both Flipkart and Amazon similar products
    flipkart_prod_data = []
    amazon_prod_data = []
    
    for idx, prod_name in enumerate(df2.product_name):
      flipkart_prod_data.append(df2.iloc[similar_prod_indexes[idx]])
      amazon_prod_data.append(df1.iloc[idx])
      #print(idx,prod_name)

    return amazon_prod_data, flipkart_prod_data

  def create_dataframe(amazon_prod_data, flipkart_prod_data):

    '''create dataframe two dataframe by on basis of similarity to data'''
    
    flipkart_prod_data = pd.DataFrame(flipkart_prod_data)
    flipkart_prod_data.columns = ['Product name in Flipkart','Retail Price in Flipkart','Discounted Price in Flipkart']
    
    amazon_prod_data = pd.DataFrame(amazon_prod_data)
    amazon_prod_data.columns = ['Product name in Amazon','Retail Price in Amazon','Discounted Price in Amazon']

    amazon_prod_data['Retail Price in Amazon'] = amazon_prod_data['Retail Price in Amazon'].astype(np.float64)
    amazon_prod_data['Discounted Price in Amazon'] = amazon_prod_data['Discounted Price in Amazon'].astype(np.float64)

    flipkart_prod_final_data = flipkart_prod_data.copy()
    flipkart_prod_final_data = flipkart_prod_final_data.reset_index(drop=True)
    flipkart_prod_final_data.head()

    return amazon_prod_data,flipkart_prod_final_data



  def export_data_csv(path_to_export,amazon_prod_data,flipkart_prod_final_data):

    '''create final database and export data in form of csv'''
    
    final_prod_data = pd.concat([flipkart_prod_final_data,amazon_prod_data],axis=1)
    final_prod_data.to_csv(path_to_export + 'final_result.csv')

    return final_prod_data

  cosine_scores = embedded_model(model_name,data1,data2)

  similar_pro_index = store_similar_prod_index(cosine_scores,data1,data2)

  similar_prod_indexes = create_sim_list(similar_pro_index)

  amazon_data,flip_data = extract_similar_data(similar_prod_indexes,data1,data2)

  amazon_df,flip_df = create_dataframe(amazon_data, flip_data)

  final_df = export_data_csv(path_to_export,amazon_df,flip_df)

  return final_df
