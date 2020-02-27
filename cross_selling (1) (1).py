import csv
import pandas as pd
import matplotlib.pyplot as plt
import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
from orangecontrib.associate.fpgrowth import *

# %matplotlib inline

"""# Construct and Load the product Dataset"""

product_items = set()
with open("item_categories.txt",encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        product_items.update(line)
output_list = list()
with open("item_categories.txt",encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        row_val = {item:0 for item in product_items}
        row_val.update({item:1 for item in line})
        output_list.append(row_val)
product_df = pd.DataFrame(output_list)
#product_df.to_csv("fe.csv")
product_df =  pd.read_csv("fe.csv")
product_df.head()

"""# View top sold items"""

total_item_id = sum(product_df.sum())
print(total_item_id)
item_summary_df = product_df.sum().sort_values(ascending = False).reset_index().head(n=20)
item_summary_df.rename(columns={item_summary_df.columns[0]:'item_category_name',item_summary_df.columns[1]:'item_id'}, inplace=True)
item_summary_df.head()

"""# Visualize top sold items"""

objects = (list(item_summary_df['item_category_name'].head(n=20)))
y_pos = np.arange(len(objects))
performance = list(item_summary_df['item_id'].head(n=20))
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Item count')
plt.title('Item sales distribution')

"""# Analyze items contributing to top sales"""

item_summary_df['item_perc'] = item_summary_df['item_id']/total_item_id
item_summary_df['total_perc'] = item_summary_df.item_perc.cumsum()
item_summary_df.head(10)

"""# Analyze items contributing to top 50% of sales"""

item_summary_df[item_summary_df.total_perc <= 0.5].shape

item_summary_df[item_summary_df.total_perc <= 0.5]

"""# Construct Orange Table"""

input_assoc_rules = product_df
domain_product = Domain([DiscreteVariable.make(name=item,values=['0', '1']) for item in input_assoc_rules.columns])
data_gro_1 = Orange.data.Table.from_numpy(domain=domain_product,  X=input_assoc_rules.as_matrix(),Y= None)

"""# Prune Dataset for frequently purchased items"""

def prune_dataset(input_df, length_trans = 2, total_sales_perc = 0.5, start_item = None, end_item = None):
    if 'total_items' in input_df.columns:
        del(input_df['total_items'])
    item_id = input_df.sum().sort_values(ascending = False).reset_index()
    total_items = sum(input_df.sum().sort_values(ascending = False))
    item_id.rename(columns={item_id.columns[0]:'item_category_name',item_id.columns[1]:'item_id'}, inplace=True)
    if not start_item and not end_item: 
        item_id['item_perc'] = item_id['item_id']/total_items
        item_id['total_perc'] = item_id.item_perc.cumsum()
        selected_items = list(item_id[item_id.total_perc < total_sales_perc].item_category_name)
        input_df['total_items'] = input_df[selected_items].sum(axis = 1)
        input_df = input_df[input_df.total_items >= length_trans]
        del(input_df['total_items'])
        return input_df[selected_items], item_id[item_id.total_perc < total_sales_perc]
    elif end_item > start_item:
        selected_items = list(item_id[start_item:end_item].item_category_name)
        input_df['total_items'] = input_df[selected_items].sum(axis = 1)
        input_df = input_df[input_df.total_items >= length_trans]
        del(input_df['total_items'])
        return input_df[selected_items],item_id[start_item:end_item]

output_df, item_ids = prune_dataset(input_df=product_df, length_trans=2,total_sales_perc=0.4)
print(output_df.shape)
print(list(output_df.columns))

"""# Association Rule Mining with FP Growth"""

input_assoc_rules = output_df
domain_product = Domain([DiscreteVariable.make(name=item,values=['0', '1']) for item in input_assoc_rules.columns])
data_gro_1 = Orange.data.Table.from_numpy(domain=domain_product,  X=input_assoc_rules.as_matrix(),Y= None)
data_gro_1_en, mapping = OneHot.encode(data_gro_1, include_class=False)

min_support = 0.01
print("num of required transactions = ", int(input_assoc_rules.shape[0]*min_support))
num_trans = input_assoc_rules.shape[0]*min_support
itemsets = dict(frequent_itemsets(data_gro_1_en, min_support=min_support))

len(itemsets)

confidence = 0.3
rules_df = pd.DataFrame()

if len(itemsets) < 1000000: 
    rules = [(P, Q, supp, conf)
    for P, Q, supp, conf in association_rules(itemsets, confidence)
       if len(Q) == 1 ]

    names = {item: '{}={}'.format(var.name, val)
        for item, var, val in OneHot.decode(mapping, data_gro_1, mapping)}
    
    eligible_ante = [v for k,v in names.items() if v.endswith("1")]
    
    N = input_assoc_rules.shape[0]
    
    rule_stats = list(rules_stats(rules, itemsets, N))
    
    rule_list_df = []
    for ex_rule_frm_rule_stat in rule_stats:
        ante = ex_rule_frm_rule_stat[0]            
        cons = ex_rule_frm_rule_stat[1]
        named_cons = names[next(iter(cons))]
        if named_cons in eligible_ante:
            rule_lhs = [names[i][:-2] for i in ante if names[i] in eligible_ante]
            ante_rule = ', '.join(rule_lhs)
            if ante_rule and len(rule_lhs)>1 :
                rule_dict = {'support' : ex_rule_frm_rule_stat[2],
                             'confidence' : ex_rule_frm_rule_stat[3],
                             'coverage' : ex_rule_frm_rule_stat[4],
                             'strength' : ex_rule_frm_rule_stat[5],
                             'lift' : ex_rule_frm_rule_stat[6],
                             'leverage' : ex_rule_frm_rule_stat[7],
                             'antecedent': ante_rule,
                             'consequent':named_cons[:-2] }
                rule_list_df.append(rule_dict)
    rules_df = pd.DataFrame(rule_list_df)
    print("Raw rules data frame of {} rules generated".format(rules_df.shape[0]))
    if not rules_df.empty:
        pruned_rules_df = rules_df.groupby(['antecedent','consequent']).max().reset_index()
    else:
        print("Unable to generate any rule")

"""# Sorting rules in our product Dataset"""

(pruned_rules_df[['antecedent','consequent',
                  'support','confidence','lift']].groupby('consequent')
                                                 .max()
                                                 .reset_index()
                                                 .sort_values(['lift', 'support','confidence'],
                                                              ascending=False))

"""# Association rule mining on our product dataset

## Load and Filter Dataset
"""

cs_mba = pd.read_excel(io=r'sales_region.xlsx')
cs_mba_Lorien = cs_mba[cs_mba.Country == 'Lorien']

cs_mba_Lorien.head()

"""Remove returned item as we are only interested in the buying patterns"""

cs_mba_Lorien = cs_mba_Lorien[~(cs_mba_Lorien.InvoiceNo.str.contains("C") == True)]
cs_mba_Lorien = cs_mba_Lorien[~cs_mba_Lorien.Quantity<0]

cs_mba_Lorien.shape

cs_mba_Lorien.InvoiceNo.value_counts().shape

"""## Build Transaction Dataset"""

items = list(cs_mba_Lorien.Description.unique())
grouped = cs_mba_Lorien.groupby('InvoiceNo')
transaction_level_df_Lorien = grouped.aggregate(lambda x: tuple(x)).reset_index()[['InvoiceNo','Description']]

transaction_dict = {item:0 for item in items}
output_dict = dict()
temp = dict()
for rec in transaction_level_df_Lorien.to_dict('records'):
    invoice_num = rec['InvoiceNo']
    items_list = rec['Description']
    transaction_dict = {item:0 for item in items}
    transaction_dict.update({item:1 for item in items if item in items_list})
    temp.update({invoice_num:transaction_dict})

new = [v for k,v in temp.items()]
tranasction_df = pd.DataFrame(new)
del(tranasction_df[tranasction_df.columns[0]])

tranasction_df.shape

tranasction_df.head()

output_df_Lorien_n, item_ids_n = prune_dataset(input_df=tranasction_df, length_trans=2, start_item=0, end_item=15)
print(output_df_Lorien_n.shape)

output_df_Lorien_n.head()

"""## Association Rule Mining with FP Growth"""

input_assoc_rules = output_df_Lorien_n
domain_transac = Domain([DiscreteVariable.make(name=item,values=['0', '1']) for item in input_assoc_rules.columns])
data_tran_Lorien = Orange.data.Table.from_numpy(domain=domain_transac,  X=input_assoc_rules.as_matrix(),Y= None)
data_tran_Lorien_en, mapping = OneHot.encode(data_tran_Lorien, include_class=True)

support = 0.01
print("num of required transactions = ", int(input_assoc_rules.shape[0]*support))
num_trans = input_assoc_rules.shape[0]*support
itemsets = dict(frequent_itemsets(data_tran_Lorien_en, support))

len(itemsets)

confidence = 0.3
rules_df = pd.DataFrame()
if len(itemsets) < 1000000: 
    rules = [(P, Q, supp, conf)
    for P, Q, supp, conf in association_rules(itemsets, confidence)
       if len(Q) == 1 ]

    names = {item: '{}={}'.format(var.name, val)
        for item, var, val in OneHot.decode(mapping, data_tran_Lorien, mapping)}
    
    eligible_ante = [v for k,v in names.items() if v.endswith("1")]
    
    N = input_assoc_rules.shape[0]
    
    rule_stats = list(rules_stats(rules, itemsets, N))
    
    rule_list_df = []
    for ex_rule_frm_rule_stat in rule_stats:
        ante = ex_rule_frm_rule_stat[0]            
        cons = ex_rule_frm_rule_stat[1]
        named_cons = names[next(iter(cons))]
        if named_cons in eligible_ante:
            rule_lhs = [names[i][:-2] for i in ante if names[i] in eligible_ante]
            ante_rule = ', '.join(rule_lhs)
            if ante_rule and len(rule_lhs)>1 :
                rule_dict = {'support' : ex_rule_frm_rule_stat[2],
                             'confidence' : ex_rule_frm_rule_stat[3],
                             'coverage' : ex_rule_frm_rule_stat[4],
                             'strength' : ex_rule_frm_rule_stat[5],
                             'lift' : ex_rule_frm_rule_stat[6],
                             'leverage' : ex_rule_frm_rule_stat[7],
                             'antecedent': ante_rule,
                             'consequent':named_cons[:-2] }
                rule_list_df.append(rule_dict)
    rules_df = pd.DataFrame(rule_list_df)
    print("Raw rules data frame of {} rules generated".format(rules_df.shape[0]))
    if not rules_df.empty:
        pruned_rules_df = rules_df.groupby(['antecedent','consequent']).max().reset_index()
    else:
        print("Unable to generate any rule")

"""## Sort and display rules"""

dw = pd.options.display.max_colwidth
pd.options.display.max_colwidth = 100
(pruned_rules_df[['antecedent','consequent',
                  'support','confidence','lift']].groupby('consequent')
                                                 .max()
                                                 .reset_index()
                                                 .sort_values(['lift', 'support','confidence'],
                                                              ascending=False)).head(5)

pd.options.display.max_colwidth = dw