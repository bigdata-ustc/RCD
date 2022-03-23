ASSIST
-
### Example to run the codes.

Go to the graph directory:
```
cd ASSIST/graph
```

Build concept dependency map:

```
python construct_concept_map.py
python process_edge.py
```

**Note**:

* We construct concept dependency map via log_data_all.json.

* We train and test model via log_data.json (remove logs that fewer than 15 response records).
