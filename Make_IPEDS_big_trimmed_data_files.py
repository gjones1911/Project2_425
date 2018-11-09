from Create_Data_files import*

# def create_needed_files(csv_name=utk_peers_csv, data_file_name=utk_data_file, json_name=utk_peers_json,
#                        imputated_data_file=imputated_data, obs_names=peer_names, attrib_names=header_names,
#                        ignore_list=ign_list, additionalignores=[0,1],verbose=True, attrib_file=attrib_file,
#                        stat_file=basic_stat_file):


IPED = 'IPEDS-big-trimmed'


create_needed_files(csv_name=IPED+'.csv', data_file_name=IPED + '.dt', json_name=IPED + '.json',
                    imputated_data_file=IPED + '-imp.dt', obs_names='Big-Schools.dt',
                    attrib_names='Big-Attrib_names.dt', ignore_list=ign_list,  additional_ignores=[], verbose=True,
                    attrib_file='Big-attributs.dt', stat_file='Big-stats.dt')
