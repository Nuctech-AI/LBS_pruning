import yaml
def get_yaml_data(yaml_file):
    file=open(yaml_file,'r',encoding='utf-8')
    file_data=file.read()
    file.close()
    data=yaml.load(file_data)
    return data