"""Read-only Parquet-footer audit. No Spark, model access or feature creation."""
import json
from pathlib import Path
from datetime import date
import pyarrow.parquet as pq

root=Path('/ods');files=sorted(root.glob('*.parquet'))
result={'ods_files':len(files),'ods_footer_rows':0,'ods_schema_variants':{},'ingestion_footer_statistics':[],
        'bounded_train_validation_ingestion_sample':None}
for p in files:
    pf=pq.ParquetFile(p);md=pf.metadata
    names=pf.schema_arrow.names;sig='|'.join(names)
    result['ods_schema_variants'][sig]=result['ods_schema_variants'].get(sig,0)+1
    result['ods_footer_rows']+=md.num_rows
    indexes={md.schema.column(i).name:i for i in range(md.num_columns)}
    for i in range(md.num_row_groups):
        rg=md.row_group(i);ing=rg.column(indexes['_ingested_at']).statistics
        entry={'file':p.name,'row_group':i,'rows':rg.num_rows,'ingested_at_statistics':str(ing)}
        result['ingestion_footer_statistics'].append(entry)
        d=rg.column(indexes['FlightDate']).statistics
        # Inspect only ingestion metadata of a source row proven within selection dates.
        if result['bounded_train_validation_ingestion_sample'] is None and d is not None and d.has_min_max and d.min>=date(2023,1,1) and d.max<=date(2024,8,31):
            row=next(pf.iter_batches(batch_size=1,row_groups=[i],columns=['FlightDate','_ingested_at','_source_file'])).to_pylist()[0]
            result['bounded_train_validation_ingestion_sample']={'file':p.name,'row_group':i,'row':row}
part=sorted(Path('/dwd-selection-month').glob('*.parquet'))[0]
pf=pq.ParquetFile(part)
result['dwd_january_2023_footer']={'file':part.name,'schema':str(pf.schema_arrow),'columns':pf.schema_arrow.names,'rows':pf.metadata.num_rows}
print(json.dumps(result,indent=2,default=str))
