def is_error(node): 
    for child_node in node.get_children(): 
      if isinstance(child_node, astroid.Raise): 
         return True 
      return False

def upgrade(migrate_engine): 
    if (migrate_engine.name != 'mysql'): 
      if (not utils.index_exists(migrate_engine, SYS_META_TABLE_NAME, INDEX_NAME)): 
         utils.add_index(migrate_engine, SYS_META_TABLE_NAME, INDEX_NAME, INDEX_COLUMNS)

def get_global_notification_type(global_subscription, user): 
    for notification_type in constants.NOTIFICATION_TYPES: 
      if getattr(global_subscription, notification_type).filter(id=user.id).exists(): 
         return notification_type

def _filter_apis(name, apis): 
    return [api for api in apis if (api['name'] == name)]

def delete(args): 
    for arg in args: 
      try: 
         wincerapi.CeDeleteFile(arg) 
         print ('Deleted:   %s' % arg) 
      except win32api.error as details: 
         print_error(details, ("Error   deleting   '%s'" % arg))

def overwriteDictionary(fromDictionary, keys, toDictionary): 
    for key in keys: 
      if (key in fromDictionary): 
         toDictionary[key] = fromDictionary[key]

def rendering_info(obj): 
    return ('<ul>%s</ul>' % ''.join((('<li>%s</li>' % (x % y)) for (x, y) in (('<img   src="%s/admin/img/admin/icon-yes.gif"   alt="%s">   Deferred   rendering', (settings.STATIC_URL, obj.defer_rendering)), ('%s   (last)', obj.last_rendered_at), ('%s   (started)', obj.render_started_at), ('%s   (scheduled)', obj.render_scheduled_at)) if y)))

def detect(code): 
    return (('   ' not in code) and (('%20' in code) or (code.count('%') > 3)))

def vm_state(vm_=None): 
    with _get_xapi_session() as xapi: 
      info = {} 
      if vm_: 
         info[vm_] = _get_record_by_label(xapi, 'VM', vm_)['power_state'] 
         return info 
      for vm_ in list_domains(): 
         info[vm_] = _get_record_by_label(xapi, 'VM', vm_)['power_state'] 
      return info

def validate_maximum(value, maximum): 
    if ((maximum is not None) and (value > maximum)): 
      raise ValueError((u'%r   must   be   smaller   than   %r.' % (value, maximum)))

def get_eol_chars_from_os_name(os_name): 
    for (eol_chars, name) in EOL_CHARS: 
      if (name == os_name): 
         return eol_chars

def do_exit(actions): 
    for action_group in actions: 
      if (len(action_group.destroy) > 0): 
         raise SystemExit(1)

def latest_version(*names, **kwargs): 
    return ('' if (len(names) == 1) else dict(((x, '') for x in names)))

def entry_choices(user, page): 
    for entry in wizard_pool.get_entries(): 
      if entry.user_has_add_permission(user, page=page): 
         (yield (entry.id, entry.title))

def _is_mobile(ntype): 
    return ((ntype == PhoneNumberType.MOBILE) or (ntype == PhoneNumberType.FIXED_LINE_OR_MOBILE) or (ntype == PhoneNumberType.PAGER))

def check_fields(context, fields): 
    for field in fields: 
      if (field.get('type') and (not _is_valid_pg_type(context, field['type']))): 
         raise ValidationError({'fields': [u'"{0}"   is   not   a   valid   field   type'.format(field['type'])]}) 
      elif (not _is_valid_field_name(field['id'])): 
         raise ValidationError({'fields': [u'"{0}"   is   not   a   valid   field   name'.format(field['id'])]})

def run(self, request, queryset): 
    if request.POST.get('_selected_action'): 
      id = request.POST.get('_selected_action') 
      siteObj = self.model.objects.get(pk=id) 
      if request.POST.get('post'): 
         for siteObj in queryset: 
            self.message_user(request, ('Executed   Backup:   ' + siteObj.name)) 
            out = StringIO.StringIO() 
            call_command('backup', force_exec=True, backup_dir=siteObj.base_folder, stdout=out) 
            value = out.getvalue() 
            if value: 
               siteObj.location = value 
               siteObj.save() 
            else: 
               self.message_user(request, (siteObj.name + '   backup   failed!')) 
      else: 
         context = {'objects_name': 'Backups', 'title': 'Confirm   run   of   Backups:', 'action_exec': 'run', 'cancellable_backups': [siteObj], 'action_checkbox_name': helpers.ACTION_CHECKBOX_NAME} 
         return TemplateResponse(request, 'admin/backups/confirm_cancel.html', context, current_app=self.admin_site.name)

def _configs_from_dir(conf_dir): 
    for filename in sorted(os.listdir(conf_dir)): 
      if (filename.startswith('.') or (not filename.endswith('.ini'))): 
         continue 
      LOG.debug(('Loading   configuration   from:   %s' % filename)) 
      try: 
         conf = configobj.ConfigObj(os.path.join(conf_dir, filename)) 
      except configobj.ConfigObjError as ex: 
         LOG.error(("Error   in   configuration   file   '%s':   %s" % (os.path.join(conf_dir, filename), ex))) 
         raise 
      conf['DEFAULT'] = dict(desktop_root=get_desktop_root(), build_dir=get_build_dir()) 
      (yield conf)

def vm_info(vm_=None): 
    with _get_xapi_session() as xapi: 
      def _info(vm_): 
         vm_rec = _get_record_by_label(xapi, 'VM', vm_) 
         if (vm_rec is False): 
            return False 
         vm_metrics_rec = _get_metrics_record(xapi, 'VM', vm_rec) 
         return {'cpu': vm_metrics_rec['VCPUs_number'], 'maxCPU': _get_val(vm_rec, ['VCPUs_max']), 'cputime': vm_metrics_rec['VCPUs_utilisation'], 'disks': get_disks(vm_), 'nics': get_nics(vm_), 'maxMem': int(_get_val(vm_rec, ['memory_dynamic_max'])), 'mem': int(vm_metrics_rec['memory_actual']), 'state': _get_val(vm_rec, ['power_state'])} 
      info = {} 
      if vm_: 
         ret = _info(vm_) 
         if (ret is not None): 
            info[vm_] = ret 
      else: 
         for vm_ in list_domains(): 
            ret = _info(vm_) 
            if (ret is not None): 
               info[vm_] = _info(vm_) 
      return info

def _validate_mutable_mappings(a, b): 
    if (not (isinstance(a, MutableMapping) and isinstance(b, MutableMapping))): 
      myvars = [] 
      for x in [a, b]: 
         try: 
            myvars.append(dumps(x)) 
         except: 
            myvars.append(to_native(x)) 
      raise AnsibleError("failed   to   combine   variables,   expected   dicts   but   got   a   '{0}'   and   a   '{1}':   \n{2}\n{3}".format(a.__class__.__name__, b.__class__.__name__, myvars[0], myvars[1]))

def add_monitor(): 
    for (name, function) in globals().items(): 
      if (not inspect.isfunction(function)): 
         continue 
      args = inspect.getargspec(function)[0] 
      if (args and name.startswith('monitor')): 
         exec ('pep8.%s   =   %s' % (name, name))

def unlink_older_than(path, mtime): 
    if os.path.exists(path): 
      for fname in listdir(path): 
         fpath = os.path.join(path, fname) 
         try: 
            if (os.path.getmtime(fpath) < mtime): 
               os.unlink(fpath) 
         except OSError: 
            pass

def test_predefined_string_roundtrip(): 
    with u.magnitude_zero_points.enable(): 
      assert (u.Unit(u.STmag.to_string()) == u.STmag) 
      assert (u.Unit(u.ABmag.to_string()) == u.ABmag) 
      assert (u.Unit(u.M_bol.to_string()) == u.M_bol) 
      assert (u.Unit(u.m_bol.to_string()) == u.m_bol)

def check_all_files(dir_path=theano.__path__[0], pattern='*.py'): 
    with open('theano_filelist.txt', 'a') as f_txt: 
      for (dir, _, files) in os.walk(dir_path): 
         for f in files: 
            if fnmatch(f, pattern): 
               error_num = flake8.main.check_file(os.path.join(dir, f), ignore=ignore) 
               if (error_num > 0): 
                  path = os.path.relpath(os.path.join(dir, f), theano.__path__[0]) 
                  f_txt.write((('"' + path) + '",\n'))

def document_generator(dir_path_pattern, count=None): 
    for (running_count, item) in enumerate(glob.iglob(dir_path_pattern)): 
      if (count and (running_count >= count)): 
         raise StopIteration() 
      doc_id = os.path.basename(item) 
      with codecs.open(item, encoding='utf-8') as f: 
         try: 
            text = f.read() 
         except UnicodeDecodeError: 
            continue 
         (yield Document(text, doc_id, item))

def query_package(module, xbps_path, name, state='present'): 
    if (state == 'present'): 
      lcmd = ('%s   %s' % (xbps_path['query'], name)) 
      (lrc, lstdout, lstderr) = module.run_command(lcmd, check_rc=False) 
      if (not is_installed(lstdout)): 
         return (False, False) 
      rcmd = ('%s   -Sun' % xbps_path['install']) 
      (rrc, rstdout, rstderr) = module.run_command(rcmd, check_rc=False) 
      if ((rrc == 0) or (rrc == 17)): 
         'Return   True   to   indicate   that   the   package   is   installed   locally,\n                                    and   the   result   of   the   version   number   comparison   to   determine   if   the\n                                    package   is   up-to-date' 
         return (True, (name not in rstdout)) 
      return (False, False)

def _record_from_json(value, field): 
    if _not_null(value, field): 
      record = {} 
      record_iter = zip(field.fields, value['f']) 
      for (subfield, cell) in record_iter: 
         converter = _CELLDATA_FROM_JSON[subfield.field_type] 
         if (subfield.mode == 'REPEATED'): 
            value = [converter(item['v'], subfield) for item in cell['v']] 
         else: 
            value = converter(cell['v'], subfield) 
         record[subfield.name] = value 
      return record

def get_class(class_string, exception=FilterError): 
    if (not hasattr(class_string, u'__bases__')): 
      try: 
         class_string = str(class_string) 
         (mod_name, class_name) = get_mod_func(class_string) 
         if class_name: 
            return getattr(__import__(mod_name, {}, {}, [str(u'')]), class_name) 
      except AttributeError as e: 
         raise exception((u'Failed   to   import   %s.   AttributeError   is:   %s' % (class_string, e))) 
      except ImportError as e: 
         raise exception((u'Failed   to   import   %s.   ImportError   is:   %s' % (class_string, e))) 
      raise exception((u"Invalid   class   path   '%s'" % class_string))

def create_tables(db, prefix, tmp_prefix): 
    for table in ('point', 'line', 'roads', 'polygon'): 
      db.execute('BEGIN') 
      try: 
         db.execute(('CREATE   TABLE   %(prefix)s_%(table)s   (   LIKE   %(tmp_prefix)s_%(table)s   )' % locals())) 
      except ProgrammingError as e: 
         db.execute('ROLLBACK') 
         if (e.pgcode != '42P07'): 
            raise 
      else: 
         db.execute(("INSERT   INTO   geometry_columns\n                                                                              (f_table_catalog,   f_table_schema,   f_table_name,   f_geometry_column,   coord_dimension,   srid,   type)\n                                                                              SELECT   f_table_catalog,   f_table_schema,   '%(prefix)s_%(table)s',   f_geometry_column,   coord_dimension,   srid,   type\n                                                                              FROM   geometry_columns   WHERE   f_table_name   =   '%(tmp_prefix)s_%(table)s'" % locals())) 
         db.execute('COMMIT')

def cron_task_host(): 
    while True: 
      if (not enable_cron_tasks): 
         if (threading.current_thread() != threading.main_thread()): 
            exit() 
         else: 
            return 
      sleep(60) 
      try: 
         task_scheduler.run() 
      except: 
         errprint('ErrorDuringExecutingCronTasks') 
         traceback.print_exc()

def _InitNinjaFlavor(params, target_list, target_dicts): 
    for qualified_target in target_list: 
      spec = target_dicts[qualified_target] 
      if spec.get('msvs_external_builder'): 
         continue 
      path_to_ninja = spec.get('msvs_path_to_ninja', 'ninja.exe') 
      spec['msvs_external_builder'] = 'ninja' 
      if (not spec.get('msvs_external_builder_out_dir')): 
         (gyp_file, _, _) = gyp.common.ParseQualifiedTarget(qualified_target) 
         gyp_dir = os.path.dirname(gyp_file) 
         configuration = '$(Configuration)' 
         if (params.get('target_arch') == 'x64'): 
            configuration += '_x64' 
         spec['msvs_external_builder_out_dir'] = os.path.join(gyp.common.RelativePath(params['options'].toplevel_dir, gyp_dir), ninja_generator.ComputeOutputDir(params), configuration) 
      if (not spec.get('msvs_external_builder_build_cmd')): 
         spec['msvs_external_builder_build_cmd'] = [path_to_ninja, '-C', '$(OutDir)', '$(ProjectName)'] 
      if (not spec.get('msvs_external_builder_clean_cmd')): 
         spec['msvs_external_builder_clean_cmd'] = [path_to_ninja, '-C', '$(OutDir)', '-tclean', '$(ProjectName)']

def distort_color(image, color_ordering=0, fast_mode=True, scope=None): 
    with tf.name_scope(scope, 'distort_color', [image]): 
      if fast_mode: 
         if (color_ordering == 0): 
            image = tf.image.random_brightness(image, max_delta=(32.0 / 255.0)) 
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5) 
         else: 
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5) 
            image = tf.image.random_brightness(image, max_delta=(32.0 / 255.0)) 
      elif (color_ordering == 0): 
         image = tf.image.random_brightness(image, max_delta=(32.0 / 255.0)) 
         image = tf.image.random_saturation(image, lower=0.5, upper=1.5) 
         image = tf.image.random_hue(image, max_delta=0.2) 
         image = tf.image.random_contrast(image, lower=0.5, upper=1.5) 
      elif (color_ordering == 1): 
         image = tf.image.random_saturation(image, lower=0.5, upper=1.5) 
         image = tf.image.random_brightness(image, max_delta=(32.0 / 255.0)) 
         image = tf.image.random_contrast(image, lower=0.5, upper=1.5) 
         image = tf.image.random_hue(image, max_delta=0.2) 
      elif (color_ordering == 2): 
         image = tf.image.random_contrast(image, lower=0.5, upper=1.5) 
         image = tf.image.random_hue(image, max_delta=0.2) 
         image = tf.image.random_brightness(image, max_delta=(32.0 / 255.0)) 
         image = tf.image.random_saturation(image, lower=0.5, upper=1.5) 
      elif (color_ordering == 3): 
         image = tf.image.random_hue(image, max_delta=0.2) 
         image = tf.image.random_saturation(image, lower=0.5, upper=1.5) 
         image = tf.image.random_contrast(image, lower=0.5, upper=1.5) 
         image = tf.image.random_brightness(image, max_delta=(32.0 / 255.0)) 
      else: 
         raise ValueError('color_ordering   must   be   in   [0,   3]') 
      return tf.clip_by_value(image, 0.0, 1.0)

def _prep_input(kwargs): 
    for kwarg in ('environment', 'lxc_conf'): 
      kwarg_value = kwargs.get(kwarg) 
      if ((kwarg_value is not None) and (not isinstance(kwarg_value, six.string_types))): 
         err = 'Invalid   {0}   configuration.   See   the   documentation   for   proper   usage.'.format(kwarg) 
         if salt.utils.is_dictlist(kwarg_value): 
            new_kwarg_value = salt.utils.repack_dictlist(kwarg_value) 
            if (not kwarg_value): 
               raise SaltInvocationError(err) 
            kwargs[kwarg] = new_kwarg_value 
         if (not isinstance(kwargs[kwarg], dict)): 
            raise SaltInvocationError(err)

def t_KEGG_Enzyme(testfiles): 
    for file in testfiles: 
      fh = open(os.path.join('KEGG', file)) 
      print((('Testing   Bio.KEGG.Enzyme   on   ' + file) + '\n\n')) 
      records = Enzyme.parse(fh) 
      for (i, record) in enumerate(records): 
         print(record) 
      fh.seek(0) 
      if (i == 0): 
         print(Enzyme.read(fh)) 
      else: 
         try: 
            print(Enzyme.read(fh)) 
            assert False 
         except ValueError as e: 
            assert (str(e) == 'More   than   one   record   found   in   handle') 
      print('\n') 
      fh.close()

def _process_worker(call_queue, result_queue, shutdown): 
    while True: 
      try: 
         call_item = call_queue.get(block=True, timeout=0.1) 
      except queue.Empty: 
         if shutdown.is_set(): 
            return 
      else: 
         try: 
            r = call_item.fn(*call_item.args, **call_item.kwargs) 
         except BaseException: 
            e = sys.exc_info()[1] 
            result_queue.put(_ResultItem(call_item.work_id, exception=e)) 
         else: 
            result_queue.put(_ResultItem(call_item.work_id, result=r))

def add_settings(mod, settings): 
    for setting in dir(mod): 
      if (not setting.isupper()): 
         continue 
      setting_value = getattr(mod, setting) 
      if ((setting in ('INSTALLED_APPS', 'TEMPLATE_DIRS')) and isinstance(setting_value, six.string_types)): 
         setting_value = (setting_value,) 
      if (setting[:6] == 'EXTRA_'): 
         base_setting = setting[6:] 
         if isinstance(getattr(settings, base_setting), (list, tuple)): 
            curval = getattr(settings, base_setting) 
            setattr(settings, base_setting, (curval + type(curval)(setting_value))) 
            continue 
      setattr(settings, setting, setting_value)

def tokenize_asdl(buf): 
    for (lineno, line) in enumerate(buf.splitlines(), 1): 
      for m in re.finditer('\\s*(\\w+|--.*|.)', line.strip()): 
         c = m.group(1) 
         if c[0].isalpha(): 
            if c[0].isupper(): 
               (yield Token(TokenKind.ConstructorId, c, lineno)) 
            else: 
               (yield Token(TokenKind.TypeId, c, lineno)) 
         elif (c[:2] == '--'): 
            break 
         else: 
            try: 
               op_kind = TokenKind.operator_table[c] 
            except KeyError: 
               raise ASDLSyntaxError(('Invalid   operator   %s' % c), lineno) 
            (yield Token(op_kind, c, lineno))

def _package_conf_file_to_dir(file_name): 
    if (file_name in SUPPORTED_CONFS): 
      path = BASE_PATH.format(file_name) 
      if os.path.exists(path): 
         if os.path.isdir(path): 
            return False 
         else: 
            os.rename(path, (path + '.tmpbak')) 
            os.mkdir(path, 493) 
            with salt.utils.fopen((path + '.tmpbak')) as fh_: 
               for line in fh_: 
                  line = line.strip() 
                  if (line and (not line.startswith('#'))): 
                     append_to_package_conf(file_name, string=line) 
            os.remove((path + '.tmpbak')) 
            return True 
      else: 
         os.mkdir(path, 493) 
         return True

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
         dict1[key] = val1

def _traverse_results(value, fields, row, path): 
    for (f, v) in value.iteritems(): 
      field_name = ('{path}.{name}'.format(path=path, name=f) if path else f) 
      if (not isinstance(v, (dict, list, tuple))): 
         if (field_name in fields): 
            row[fields.index(field_name)] = ensure_utf(v) 
      elif (isinstance(v, dict) and (f != 'attributes')): 
         _traverse_results(v, fields, row, field_name)

def coerce_dtypes(df, dtypes): 
    for c in df.columns: 
      if ((c in dtypes) and (df.dtypes[c] != dtypes[c])): 
         if (np.issubdtype(df.dtypes[c], np.floating) and np.issubdtype(dtypes[c], np.integer)): 
            if (df[c] % 1).any(): 
               msg = "Runtime   type   mismatch.   Add   {'%s':   float}   to   dtype=   keyword   in   read_csv/read_table" 
               raise TypeError((msg % c)) 
         df[c] = df[c].astype(dtypes[c])

def traverse(roots, parent='', verbose=False): 
    for root in roots: 
      if root.method_map: 
         print('->', ((parent + '/') + root.raw_segment)) 
         if verbose: 
            for (method, func) in root.method_map.items(): 
               if (func.__name__ != 'method_not_allowed'): 
                  print('-->{0}   {1}:{2}'.format(method, inspect.getsourcefile(func), inspect.getsourcelines(func)[1])) 
      if root.children: 
         traverse(root.children, ((parent + '/') + root.raw_segment), verbose)

def user_details(strategy, details, user=None, *args, **kwargs): 
    if user: 
      changed = False 
      protected = (('username', 'id', 'pk', 'email') + tuple(strategy.setting('PROTECTED_USER_FIELDS', []))) 
      for (name, value) in details.items(): 
         if (not hasattr(user, name)): 
            continue 
         current_value = getattr(user, name, None) 
         if ((not current_value) or (name not in protected)): 
            changed |= (current_value != value) 
            setattr(user, name, value) 
      if changed: 
         strategy.storage.user.changed(user)

def _check_children(node): 
    for child in node.get_children(): 
      ok = False 
      if (child is None): 
         print(('Hm,   child   of   %s   is   None' % node)) 
         continue 
      if (not hasattr(child, 'parent')): 
         print(('   ERROR:   %s   has   child   %s   %x   with   no   parent' % (node, child, id(child)))) 
      elif (not child.parent): 
         print(('   ERROR:   %s   has   child   %s   %x   with   parent   %r' % (node, child, id(child), child.parent))) 
      elif (child.parent is not node): 
         print(('   ERROR:   %s   %x   has   child   %s   %x   with   wrong   parent   %s' % (node, id(node), child, id(child), child.parent))) 
      else: 
         ok = True 
      if (not ok): 
         print('lines;', node.lineno, child.lineno) 
         print('of   module', node.root(), node.root().name) 
         raise AstroidBuildingException 
      _check_children(child)

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
      (yield item.item)

