# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

number_of_composition = 3  # 成分の数
preprocess_name = 'C'  # 'raw' or 'C' or 'H'   モノマー構造の前処理

import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

no_meaning_number = -999
dataset = pd.read_csv('smiles_composition.csv', encoding='SHIFT-JIS', index_col=0)  # SMILES 付きデータセットの読み込み
raw_smiles = dataset.iloc[:, :number_of_composition]  # 分子の SMILES と組成比
compositions = dataset.iloc[:, number_of_composition:(number_of_composition * 2)]  # 分子の SMILES と組成比

all_smiles = pd.DataFrame([])
for i in range(number_of_composition):
    all_smiles = pd.concat([all_smiles, raw_smiles.iloc[:, i].dropna()])
all_smiles = all_smiles.loc[~all_smiles.iloc[:, 0].duplicated(keep='first'),:]  #重複したサンプルの削除
all_smiles.reset_index(inplace=True, drop=True)
all_smiles.columns = ['SMILES']

raw_all_smiles = all_smiles.iloc[:, 0].copy()
if preprocess_name == 'raw':
    print('SMILES の変換は行われません')
elif preprocess_name == 'C':
    for i in range(all_smiles.shape[0]):
        all_smiles.iloc[i, 0] = all_smiles.iloc[i, 0].replace('[*]', 'C')
elif preprocess_name == 'H':
    for i in range(all_smiles.shape[0]):
        all_smiles.iloc[i, 0] = all_smiles.iloc[i, 0].replace('[*]', '')
        all_smiles.iloc[i, 0] = all_smiles.iloc[i, 0].replace('()', '')
else:
    sys.exit('preprocess_name が異なります')

all_smiles = all_smiles.iloc[:, 0]

# 計算する記述子名の取得
descriptor_names = []
for descriptor_information in Descriptors.descList:
    descriptor_names.append(descriptor_information[0])

# SMILES からの記述子計算
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
raw_descriptors = []  # ここに計算された記述子の値を追加
#print('分子の数 :', len(all_smiles))
for index, smiles_i in enumerate(all_smiles):
#    print(index + 1, '/', len(all_smiles))
    molecule = Chem.MolFromSmiles(smiles_i)
    raw_descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
raw_descriptors = pd.DataFrame(raw_descriptors, index=all_smiles.index, columns=descriptor_names)
#raw_descriptors = raw_descriptors.drop(['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge'], axis=1)

# 共重合用の記述子計算
raw_smiles = raw_smiles.fillna(no_meaning_number)
descriptors = []  # ここに計算された記述子の値を追加
for sample in range(raw_smiles.shape[0]):
#    print(sample + 1, '/', raw_smiles.shape[0])
    descriptor = np.zeros(raw_descriptors.shape[1])
    for component in range(raw_smiles.shape[1]):
        if raw_smiles.iloc[sample, component] != no_meaning_number:
            descriptor += np.ndarray.flatten(raw_descriptors.loc[raw_all_smiles == raw_smiles.iloc[sample, component]].values) * compositions.iloc[sample, component]
    descriptors.append(descriptor)

descriptors = pd.DataFrame(descriptors, index=compositions.index, columns=raw_descriptors.columns)
descriptors.to_csv('descriptors_{0}.csv'.format(preprocess_name))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
