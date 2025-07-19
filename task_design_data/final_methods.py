import pylibsrcml
#Number of random runs: 11
##Easy
def is_error(node): 
    for child_node in node.get_children(): 
      if isinstance(child_node, astroid.Raise): 
         return True 
      return False #
    
def _filter_apis(name, apis): 
    return [api for api in apis if (api['name'] == name)] 

def overwriteDictionary(fromDictionary, keys, toDictionary): 
    for key in keys: 
      if (key in fromDictionary): 
         toDictionary[key] = fromDictionary[key] 

def detect(code): 
    return (('   ' not in code) and (('%20' in code) or (code.count('%') > 3))) 

def validate_maximum(value, maximum): 
    if ((maximum is not None) and (value > maximum)): 
      raise ValueError((u'%r   must   be   smaller   than   %r.' % (value, maximum))) 
    
def do_exit(actions): 
    for action_group in actions: 
      if (len(action_group.destroy) > 0): 
         raise SystemExit(1) 
      
def dict_delall(d, keys): 
    for key in keys: 
      try: 
         del d[key] 
      except KeyError: 
         pass      

def GetChild(node, tag): 
    for child in node.getchildren(): 
      if (GetTag(child) == tag): 
         return child 
      
def get_metadata(headers): 
    return dict(((k, v) for (k, v) in headers.iteritems() if k.startswith('x-goog-meta-'))) 

def get_imlist(path): 
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')] 

def removeCSVFile(csvFilePath): 
    if (('alterations' in csvFilePath) and ('example_' not in csvFilePath)): 
      os.remove(csvFilePath) 
      print ('removeGeneratedFiles   deleted   ' + csvFilePath) ###

####### Medium  

def unlink_older_than(path, mtime): 
    if os.path.exists(path): 
      for fname in listdir(path): 
         fpath = os.path.join(path, fname) 
         try: 
            if (os.path.getmtime(fpath) < mtime): 
               os.unlink(fpath) 
         except OSError: 
            pass 

def _normalize_configuration_objs(configurations): 
    for c in configurations: 
      if (not hasattr(c, 'properties')): 
         c.properties = [] 
      if hasattr(c, 'configurations'): 
         if (not c.configurations): 
            del c.configurations 
         else: 
            _normalize_configuration_objs(c.configurations) 

def _validate_min_score(min_score): 
    if min_score: 
      message = (_('%(min_score)s   is   not   a   valid   grade   percentage') % {'min_score': min_score}) 
      try: 
         min_score = int(min_score) 
      except ValueError: 
         raise GatingValidationError(message) 
      if ((min_score < 0) or (min_score > 100)): 
         raise GatingValidationError(message) 
      
def mkdirs(path): 
    if (not os.path.isdir(path)): 
      try: 
         os.makedirs(path) 
      except OSError as err: 
         if ((err.errno != errno.EEXIST) or (not os.path.isdir(path))): 
            raise
         
def get_numpy_dtype(obj): 
    if (ndarray is not FakeObject): 
      import numpy as np 
      if (isinstance(obj, np.generic) or isinstance(obj, np.ndarray)): 
         try: 
            return obj.dtype.type 
         except (AttributeError, RuntimeError): 
            return
         
def check_abstract_methods(base, subclass): 
    for attrname in dir(base): 
      if attrname.startswith('_'): 
         continue 
      attr = getattr(base, attrname) 
      if is_abstract_method(attr): 
         oattr = getattr(subclass, attrname) 
         if is_abstract_method(oattr): 
            raise Exception(('%s.%s   not   overridden' % (subclass.__name__, attrname)))
         
def print_results(distributions, list_all_files): 
    for dist in distributions: 
      logger.notify('---') 
      logger.notify(('Name:   %s' % dist['name'])) 
      logger.notify(('Version:   %s' % dist['version'])) 
      logger.notify(('Location:   %s' % dist['location'])) 
      logger.notify(('Requires:   %s' % ',   '.join(dist['requires']))) 
      if list_all_files: 
         logger.notify('Files:') 
         if ('files' in dist): 
            for line in open(dist['files']): 
               logger.notify(('      %s' % line.strip())) 
         else: 
            logger.notify('Cannot   locate   installed-files.txt')

def _keysFromFilepaths(filepaths, parseKey): 
    for fp in filepaths: 
      if fp.exists(): 
         try: 
            with fp.open() as f: 
               for key in readAuthorizedKeyFile(f, parseKey): 
                  (yield key) 
         except (IOError, OSError) as e: 
            log.msg('Unable   to   read   {0}:   {1!s}'.format(fp.path, e))

def add(repo='.', paths=None): 
    with open_repo_closing(repo) as r: 
      if (not paths): 
         paths = [] 
         for (dirpath, dirnames, filenames) in os.walk(r.path): 
            if ('.git' in dirnames): 
               dirnames.remove('.git') 
            for filename in filenames: 
               paths.append(os.path.join(dirpath[(len(r.path) + 1):], filename)) 
      r.stage(paths)

def copy_tcltk(src, dest, symlink): 
    for libversion in ('8.5', '8.6'): 
      for libname in ('tcl', 'tk'): 
         srcdir = join(src, 'tcl', (libname + libversion)) 
         destdir = join(dest, 'tcl', (libname + libversion)) 
         if (os.path.exists(srcdir) and (not os.path.exists(destdir))): 
            copyfileordir(srcdir, destdir, symlink)

def ensure_sys_path_contains(paths): 
    for entry in paths: 
      if isinstance(entry, (list, tuple)): 
         ensure_sys_path_contains(entry) 
      elif ((entry is not None) and (entry not in sys.path)): 
         sys.path.append(entry)

### Hard 
def merge(dict1, dict2): 
    for (key, val2) in dict2.items(): 
      if (val2 is not None): 
         val1 = dict1.get(key) 
         if isinstance(val2, dict): 
            if (val1 is None): 
               val1 = {} 
            if isinstance(val1, Alias): 
               val1 = (val1, val2) 
            elif isinstance(val1, tuple): 
               (alias, others) = val1 
               others = others.copy() 
               merge(others, val2) 
               val1 = (alias, others) 
            else: 
               val1 = val1.copy() 
               merge(val1, val2) 
         else: 
            val1 = val2 
         dict1[key] = val1 ###


def _traverse_results(value, fields, row, path): 
    for (f, v) in value.iteritems(): 
      field_name = ('{path}.{name}'.format(path=path, name=f) if path else f) 
      if (not isinstance(v, (dict, list, tuple))): 
         if (field_name in fields): 
            row[fields.index(field_name)] = ensure_utf(v) 
      elif (isinstance(v, dict) and (f != 'attributes')): 
         _traverse_results(v, fields, row, field_name) ### hard

def consume_queue(queue, cascade_stop): 
    while True: 
      try: 
         item = queue.get(timeout=0.1) 
      except Empty: 
         (yield None) 
         continue 
      except thread.error: 
         raise ShutdownException() 
      if item.exc: 
         raise item.exc 
      if item.is_stop: 
         if cascade_stop: 
            raise StopIteration 
         else: 
            continue 
      (yield item.item) ####

def recursive_update_dict(root, changes, ignores=()): 
    if isinstance(changes, dict): 
      for (k, v) in changes.items(): 
         if isinstance(v, dict): 
            if (k not in root): 
               root[k] = {} 
            recursive_update_dict(root[k], v, ignores) 
         elif (v in ignores): 
            if (k in root): 
               root.pop(k) 
         else: 
            root[k] = v 
def get_value_from_json(json_dict, sensor_type, group, tool): 
    if (group in json_dict): 
      if (sensor_type in json_dict[group]): 
         if ((sensor_type == 'target') and (json_dict[sensor_type] is None)): 
            return 0 
         else: 
            return json_dict[group][sensor_type] 
      elif (tool is not None): 
         if (sensor_type in json_dict[group][tool]): 
            return json_dict[group][tool][sensor_type]

def GetJavaJars(target_list, target_dicts, toplevel_dir): 
    for target_name in target_list: 
      target = target_dicts[target_name] 
      for action in target.get('actions', []): 
         for input_ in action['inputs']: 
            if ((os.path.splitext(input_)[1] == '.jar') and (not input_.startswith('$'))): 
               if os.path.isabs(input_): 
                  (yield input_) 
               else: 
                  (yield os.path.join(os.path.dirname(target_name), input_))
                  
def RemoveSelfDependencies(targets): 
    for (target_name, target_dict) in targets.iteritems(): 
      for dependency_key in dependency_sections: 
         dependencies = target_dict.get(dependency_key, []) 
         if dependencies: 
            for t in dependencies: 
               if (t == target_name): 
                  if targets[t].get('variables', {}).get('prune_self_dependency', 0): 
                     target_dict[dependency_key] = Filter(dependencies, target_name)

def parse_assigned_metadata(csvfile): 
    with open(csvfile, 'rb') as f: 
      for record in csv.reader(f): 
         module = record[0] 
         supported_by = record[1] 
         status = [] 
         if record[2]: 
            status.append('stableinterface') 
         if record[4]: 
            status.append('deprecated') 
         if record[5]: 
            status.append('removed') 
         if record[6]: 
            status.append('tested') 
         if ((not status) or record[3]): 
            status.append('preview') 
         (yield (module, {'version': '1.0', 'supported_by': supported_by, 'status': status}))

def test_client_options(config): 
    if config['use_ssl']: 
      if (('certificate' in config) and config['certificate']): 
         read_file(config['certificate']) 
      if (('client_cert' in config) and config['client_cert']): 
         read_file(config['client_cert']) 
      if (('client_key' in config) and config['client_key']): 
         read_file(config['client_key'])

def collect(names, match=(lambda name: True), match_dir=(lambda name: True)): 
    for name in names: 
      if os.path.isdir(name): 
         for (root, dirs, filenames) in os.walk(name): 
            dirs[:] = [dir for dir in dirs if match_dir(dir)] 
            for filename in filenames: 
               if match(filename): 
                  (yield os.path.join(root, filename)) 
      else: 
         (yield name)

def update_csp(): 
    for key in ('CSP_SCRIPT_SRC',): 
      values = getattr(settings, key) 
      new = set() 
      for value in values: 
         if (value.startswith('https://') and settings.DEBUG): 
            res = value.replace('https://', 'http://') 
            for v in (value, res): 
               new.add(v) 
            continue 
         elif (value.startswith('http://') and (not settings.DEBUG)): 
            continue 
         else: 
            new.add(value) 
      setattr(settings, key, tuple(new))