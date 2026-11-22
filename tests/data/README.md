# Test data

`etl_jalisco_v1.csv` is a verbatim copy of `geodataops-etl/output/municipios_indicadores.csv`
(ETL commit `ac0b177`, SHA-256 `462082b0cc34…`), so the container smoke test can mount the
real ETL output without the ETL repository being present.

It holds 18 real census observations (total, female and male population) for Guadalajara
(`14039`), San Pedro Tlaquepaque (`14098`) and Zapopan (`14120`) in 2010 and 2020, transcribed by
the ETL from IIEG with INEGI data (Censo de Población y Vivienda 2010 y 2020). See
`geodataops-etl/data/jalisco/README.md` for the source table and attribution.
