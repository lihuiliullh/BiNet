For MetaQA 2-hop and 3-hop, just need to generate new triples according to the python file, and add the these triples to query file.

For MetaQA 1-hop, generate kg_adj_map use python file, the run the code. (Can use ComplEx complete KG, the accuracy will improve about 0.2% to 0.4%).

When generate kg_adj_map, first run for_meta_1hop_complete_KG.py then run MetaQA1hop_xx_kgMap_generate.py


