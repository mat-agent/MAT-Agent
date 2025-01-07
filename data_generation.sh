#1. query, file, and traj generation
python -m data_generation.gaia_pipeline.0_query_generation_tonggpt --timestamp 20241223-213646
python -m data_generation.gaia_pipeline.1_query2file_content_parallel_tonggpt --timestamp 20241223-213646
python -m data_generation.gaia_pipeline.2_file_content2file_tonggpt --timestamp 20241223-213646  --start 0 --end 1000 
python -m data_generation.gaia_pipeline.3_traj_genetation_tonggpt --timestamp 20241223-213646  --start 0 --end 1000 

#2. verification and structure conversation
python -m data_generation.gaia_pipeline.verifier.0_collect --timestamp 20241223-213646
python -m data_generation.gaia_pipeline.verifier.1_gaia_q_f_filter --timestamp 20241223-213646
python -m data_generation.gaia_pipeline.verifier.2_convert_format --timestamp 20241223-213646
python -m data_generation.gaia_pipeline.verifier.3_gaia_verifier_parallel --timestamp 20241223-213646
