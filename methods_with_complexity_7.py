def problem_rheader(r, tabs=[]): 
    if (r.representation == 'html'): 
      if (r.record is None): 
         return None 
      problem = r.record 
      tabs = [(T('Problems'), 'problems'), (T('Solutions'), 'solution'), (T('Discuss'), 'discuss'), (T('Vote'), 'vote'), (T('Scale   of   Results'), 'results')] 
      duser = s3db.delphi_DelphiUser(problem.group_id) 
      if duser.authorised: 
         tabs.append((T('Edit'), None)) 
      rheader_tabs = s3_rheader_tabs(r, tabs) 
      rtable = TABLE(TR(TH(('%s:   ' % T('Problem'))), problem.name, TH(('%s:   ' % T('Active'))), problem.active), TR(TH(('%s:   ' % T('Description'))), problem.description), TR(TH(('%s:   ' % T('Criteria'))), problem.criteria)) 
      if (r.component and (r.component_name == 'solution') and r.component_id): 
         stable = s3db.delphi_solution 
         query = (stable.id == r.component_id) 
         solution = db(query).select(stable.name, stable.description, limitby=(0, 1)).first() 
         rtable.append(DIV(TR(TH(('%s:   ' % T('Solution'))), solution.name), TR(TH(('%s:   ' % T('Description'))), solution.description))) 
      rheader = DIV(rtable, rheader_tabs) 
      return rheader

def get_collectors_from_module(mod): 
    for attrname in dir(mod): 
      attr = getattr(mod, attrname) 
      if (inspect.isclass(attr) and issubclass(attr, Collector) and (attr != Collector)): 
         if attrname.startswith('parent_'): 
            continue 
         fqcn = '.'.join([mod.__name__, attrname]) 
         try: 
            cls = load_dynamic_class(fqcn, Collector) 
            (yield (cls.__name__, cls)) 
         except Exception: 
            logger.error('Failed   to   load   Collector:   %s.   %s', fqcn, traceback.format_exc()) 
            continue

def get_class_alias(klass): 
    for (k, v) in pyamf.ALIAS_TYPES.iteritems(): 
      for kl in v: 
         try: 
            if issubclass(klass, kl): 
               return k 
         except TypeError: 
            if hasattr(kl, '__call__'): 
               if (kl(klass) is True): 
                  return k

def _traverse_results(value, fields, row, path): 
    for (f, v) in value.iteritems(): 
      field_name = ('{path}.{name}'.format(path=path, name=f) if path else f) 
      if (not isinstance(v, (dict, list, tuple))): 
         if (field_name in fields): 
            row[fields.index(field_name)] = ensure_utf(v) 
      elif (isinstance(v, dict) and (f != 'attributes')): 
         _traverse_results(v, fields, row, field_name)

def dead_code_elimination(graph, du, ud): 
    for node in graph.rpo: 
      for (i, ins) in node.get_loc_with_ins()[:]: 
         reg = ins.get_lhs() 
         if (reg is not None): 
            if ((reg, i) not in du): 
               if ins.is_call(): 
                  ins.remove_defined_var() 
               elif ins.has_side_effect(): 
                  continue 
               else: 
                  update_chain(graph, i, du, ud) 
                  graph.remove_ins(i)

def params_to_incoming(incoming, inputs, input_values, app, name_prefix=''): 
    for input in inputs.values(): 
      if (isinstance(input, Repeat) or isinstance(input, UploadDataset)): 
         for d in input_values[input.name]: 
            index = d['__index__'] 
            new_name_prefix = (name_prefix + ('%s_%d|' % (input.name, index))) 
            params_to_incoming(incoming, input.inputs, d, app, new_name_prefix) 
      elif isinstance(input, Conditional): 
         values = input_values[input.name] 
         current = values['__current_case__'] 
         new_name_prefix = ((name_prefix + input.name) + '|') 
         incoming[(new_name_prefix + input.test_param.name)] = values[input.test_param.name] 
         params_to_incoming(incoming, input.cases[current].inputs, values, app, new_name_prefix) 
      elif isinstance(input, Section): 
         values = input_values[input.name] 
         new_name_prefix = ((name_prefix + input.name) + '|') 
         params_to_incoming(incoming, input.inputs, values, app, new_name_prefix) 
      else: 
         value = input_values.get(input.name) 
         incoming[(name_prefix + input.name)] = value

def cocktail_shaker_sort(unsorted): 
    for i in range((len(unsorted) - 1), 0, (-1)): 
      swapped = False 
      for j in range(i, 0, (-1)): 
         if (unsorted[j] < unsorted[(j - 1)]): 
            (unsorted[j], unsorted[(j - 1)]) = (unsorted[(j - 1)], unsorted[j]) 
            swapped = True 
      for j in range(i): 
         if (unsorted[j] > unsorted[(j + 1)]): 
            (unsorted[j], unsorted[(j + 1)]) = (unsorted[(j + 1)], unsorted[j]) 
            swapped = True 
      if (not swapped): 
         return unsorted

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

def vbd_unplug_with_retry(session, vbd): 
    while True: 
      try: 
         session.xenapi.VBD.unplug(vbd) 
         logging.debug(_('VBD.unplug   successful   first   time.')) 
         return 
      except XenAPI.Failure as e: 
         if ((len(e.details) > 0) and (e.details[0] == 'DEVICE_DETACH_REJECTED')): 
            logging.debug(_('VBD.unplug   rejected:   retrying...')) 
            time.sleep(1) 
         elif ((len(e.details) > 0) and (e.details[0] == 'DEVICE_ALREADY_DETACHED')): 
            logging.debug(_('VBD.unplug   successful   eventually.')) 
            return 
         else: 
            logging.error(_('Ignoring   XenAPI.Failure   in   VBD.unplug:   %s'), e) 
            return

def parse_redaction_policy_from_file(filename): 
    with open(filename) as f: 
      s = f.read().strip() 
      if (not s): 
         return RedactionPolicy([]) 
      scheme = json.loads(s) 
      try: 
         version = str(scheme.pop('version')) 
      except KeyError: 
         raise ValueError('Redaction   policy   is   missing   `version`   field') 
      if (version != '1'): 
         raise ValueError(('unknown   version   `%s`' % version)) 
      try: 
         rules = scheme.pop('rules') 
      except KeyError: 
         raise ValueError('Redaction   policy   is   missing   `rules`   field') 
      rules = [parse_one_rule_from_dict(rule) for rule in rules] 
      if scheme: 
         raise ValueError(('Redaction   policy   contains   unknown   field(s):   %s' % scheme.keys())) 
      return RedactionPolicy(rules)

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

def nova_docstring_multiline_start(physical_line, previous_logical, tokens): 
    if is_docstring(physical_line, previous_logical): 
      pos = max([physical_line.find(i) for i in START_DOCSTRING_TRIPLE]) 
      if ((len(tokens) == 0) and (pos != (-1)) and (len(physical_line) == (pos + 4))): 
         if (physical_line.strip() in START_DOCSTRING_TRIPLE): 
            return (pos, 'N404:   multi   line   docstring   should   start   with   a   summary')

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

def pollProcess(process, suppress_errors=False): 
    while True: 
      dataToStdout('.') 
      time.sleep(1) 
      returncode = process.poll() 
      if (returncode is not None): 
         if (not suppress_errors): 
            if (returncode == 0): 
               dataToStdout('   done\n') 
            elif (returncode < 0): 
               dataToStdout(('   process   terminated   by   signal   %d\n' % returncode)) 
            elif (returncode > 0): 
               dataToStdout(('   quit   unexpectedly   with   return   code   %d\n' % returncode)) 
         break

def _pick_runner_opts(runner_alias=None, cloud_role=None): 
    return set((opt_name for (opt_name, conf) in _RUNNER_OPTS.items() if (((runner_alias is None) or (conf.get('runners') is None) or (runner_alias in conf['runners'])) and ((cloud_role is None) or (cloud_role == conf.get('cloud_role'))))))

def expand_tokens(tokens, equal=False): 
    for token in tokens: 
      for pre in token.pre_tags: 
         (yield pre) 
      if ((not equal) or (not token.hide_when_equal)): 
         if token.trailing_whitespace: 
            (yield (token.html() + token.trailing_whitespace)) 
         else: 
            (yield token.html()) 
      for post in token.post_tags: 
         (yield post)

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

def eventloop(conn, limit=None, timeout=None, ignore_timeouts=False): 
    for i in ((limit and range(limit)) or count()): 
      try: 
         (yield conn.drain_events(timeout=timeout)) 
      except socket.timeout: 
         if (timeout and (not ignore_timeouts)): 
            raise

def RemoveLinkDependenciesFromNoneTargets(targets): 
    for (target_name, target_dict) in targets.iteritems(): 
      for dependency_key in dependency_sections: 
         dependencies = target_dict.get(dependency_key, []) 
         if dependencies: 
            for t in dependencies: 
               if (target_dict.get('type', None) == 'none'): 
                  if targets[t].get('variables', {}).get('link_dependency', 0): 
                     target_dict[dependency_key] = Filter(target_dict[dependency_key], t)

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

def _propagate_internal_output(graph, node, field, connections, portinputs): 
    for (destnode, inport, src) in connections: 
      if (field in portinputs): 
         (srcnode, srcport) = portinputs[field] 
         if (isinstance(srcport, tuple) and isinstance(src, tuple)): 
            src_func = srcport[1].split(u'\\n')[0] 
            dst_func = src[1].split(u'\\n')[0] 
            raise ValueError(u"Does   not   support   two   inline   functions   in   series   ('{}'      and   '{}'),   found   when   connecting   {}   to   {}.   Please   use   a   Function   node.".format(src_func, dst_func, srcnode, destnode)) 
         connect = graph.get_edge_data(srcnode, destnode, default={u'connect': []}) 
         if isinstance(src, tuple): 
            connect[u'connect'].append(((srcport, src[1], src[2]), inport)) 
         else: 
            connect = {u'connect': [(srcport, inport)]} 
         old_connect = graph.get_edge_data(srcnode, destnode, default={u'connect': []}) 
         old_connect[u'connect'] += connect[u'connect'] 
         graph.add_edges_from([(srcnode, destnode, old_connect)]) 
      else: 
         value = getattr(node.inputs, field) 
         if isinstance(src, tuple): 
            value = evaluate_connect_function(src[1], src[2], value) 
         destnode.set_input(inport, value)

def resnet_v1(inputs, blocks, num_classes=None, is_training=True, global_pool=True, output_stride=None, include_root_block=True, reuse=None, scope=None): 
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc: 
      end_points_collection = (sc.name + '_end_points') 
      with slim.arg_scope([slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense], outputs_collections=end_points_collection): 
         with slim.arg_scope([slim.batch_norm], is_training=is_training): 
            net = inputs 
            if include_root_block: 
               if (output_stride is not None): 
                  if ((output_stride % 4) != 0): 
                     raise ValueError('The   output_stride   needs   to   be   a   multiple   of   4.') 
                  output_stride /= 4 
               net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1') 
               net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1') 
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride) 
            if global_pool: 
               net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True) 
            if (num_classes is not None): 
               net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits') 
            end_points = slim.utils.convert_collection_to_dict(end_points_collection) 
            if (num_classes is not None): 
               end_points['predictions'] = slim.softmax(net, scope='predictions') 
            return (net, end_points)

def test_fix_types(): 
    for (fname, change) in ((hp_fif_fname, True), (test_fif_fname, False), (ctf_fname, False)): 
      raw = read_raw_fif(fname) 
      mag_picks = pick_types(raw.info, meg='mag') 
      other_picks = np.setdiff1d(np.arange(len(raw.ch_names)), mag_picks) 
      if change: 
         for ii in mag_picks: 
            raw.info['chs'][ii]['coil_type'] = FIFF.FIFFV_COIL_VV_MAG_T2 
      orig_types = np.array([ch['coil_type'] for ch in raw.info['chs']]) 
      raw.fix_mag_coil_types() 
      new_types = np.array([ch['coil_type'] for ch in raw.info['chs']]) 
      if (not change): 
         assert_array_equal(orig_types, new_types) 
      else: 
         assert_array_equal(orig_types[other_picks], new_types[other_picks]) 
         assert_true((orig_types[mag_picks] != new_types[mag_picks]).all()) 
         assert_true((new_types[mag_picks] == FIFF.FIFFV_COIL_VV_MAG_T3).all())

def daemonize(enable_stdio_inheritance=False): 
    if ('GUNICORN_FD' not in os.environ): 
      if os.fork(): 
         os._exit(0) 
      os.setsid() 
      if os.fork(): 
         os._exit(0) 
      os.umask(18) 
      if (not enable_stdio_inheritance): 
         closerange(0, 3) 
         fd_null = os.open(REDIRECT_TO, os.O_RDWR) 
         if (fd_null != 0): 
            os.dup2(fd_null, 0) 
         os.dup2(fd_null, 1) 
         os.dup2(fd_null, 2) 
      else: 
         fd_null = os.open(REDIRECT_TO, os.O_RDWR) 
         if (fd_null != 0): 
            os.close(0) 
            os.dup2(fd_null, 0) 
         def redirect(stream, fd_expect): 
            try: 
               fd = stream.fileno() 
               if ((fd == fd_expect) and stream.isatty()): 
                  os.close(fd) 
                  os.dup2(fd_null, fd) 
            except AttributeError: 
               pass 
         redirect(sys.stdout, 1) 
         redirect(sys.stderr, 2)

def launch(__INSTANCE__=None, **kw): 
    for (k, v) in kw.iteritems(): 
      if (v is True): 
         v = k 
         k = '' 
      try: 
         v = int(v) 
      except: 
         old = v 
         v = logging.DEBUG 
         def dofail(): 
            core.getLogger(k).error('Bad   log   level:   %s.   Defaulting   to   DEBUG.', old) 
         if ((len(old) == 0) or (len(old.strip(string.ascii_uppercase)) != 0)): 
            dofail() 
         else: 
            vv = getattr(logging, old, None) 
            if (not isinstance(vv, int)): 
               dofail() 
            else: 
               v = vv 
      core.getLogger(k).setLevel(v)

def _collect_post_update_commands(base_mapper, uowtransaction, table, states_to_update, post_update_cols): 
    for (state, state_dict, mapper, connection) in states_to_update: 
      pks = mapper._pks_by_table[table] 
      params = {} 
      hasdata = False 
      for col in mapper._cols_by_table[table]: 
         if (col in pks): 
            params[col._label] = mapper._get_state_attr_by_column(state, state_dict, col, passive=attributes.PASSIVE_OFF) 
         elif (col in post_update_cols): 
            prop = mapper._columntoproperty[col] 
            history = state.manager[prop.key].impl.get_history(state, state_dict, attributes.PASSIVE_NO_INITIALIZE) 
            if history.added: 
               value = history.added[0] 
               params[col.key] = value 
               hasdata = True 
      if hasdata: 
         (yield (params, connection))

def run_hook(component, translation, script, env=None, *args): 
    if script: 
      command = [script] 
      if args: 
         command.extend(args) 
      if component.is_repo_link: 
         target = component.linked_subproject 
      else: 
         target = component 
      environment = {'WL_VCS': target.vcs, 'WL_REPO': target.repo, 'WL_PATH': target.get_path(), 'WL_FILEMASK': component.filemask, 'WL_TEMPLATE': component.template, 'WL_FILE_FORMAT': component.file_format, 'WL_BRANCH': component.branch} 
      if translation: 
         environment['WL_LANGUAGE'] = translation.language_code 
      if (env is not None): 
         environment.update(env) 
      try: 
         subprocess.check_call(command, env=get_clean_env(environment), cwd=component.get_path()) 
         return True 
      except (OSError, subprocess.CalledProcessError) as err: 
         component.log_error('failed   to   run   hook   script   %s:   %s', script, err) 
         return False

def notify_unreplied(): 
    for email_account in frappe.get_all(u'Email   Account', u'name', filters={u'enable_incoming': 1, u'notify_if_unreplied': 1}): 
      email_account = frappe.get_doc(u'Email   Account', email_account.name) 
      if email_account.append_to: 
         for comm in frappe.get_all(u'Communication', u'name', filters={u'sent_or_received': u'Received', u'reference_doctype': email_account.append_to, u'unread_notification_sent': 0, u'email_account': email_account.name, u'creation': (u'<', (datetime.now() - timedelta(seconds=((email_account.unreplied_for_mins or 30) * 60)))), u'creation': (u'>', (datetime.now() - timedelta(seconds=(((email_account.unreplied_for_mins or 30) * 60) * 3))))}): 
            comm = frappe.get_doc(u'Communication', comm.name) 
            if (frappe.db.get_value(comm.reference_doctype, comm.reference_name, u'status') == u'Open'): 
               frappe.sendmail(recipients=email_account.get_unreplied_notification_emails(), content=comm.content, subject=comm.subject, doctype=comm.reference_doctype, name=comm.reference_name) 
            comm.db_set(u'unread_notification_sent', 1)

def fixup_for_packaged(): 
    if exists(join(ROOT, 'PKG-INFOvi   ')): 
      if (('--build-js' in sys.argv) or ('--install-js' in sys.argv)): 
         print(SDIST_BUILD_WARNING) 
         if ('--build-js' in sys.argv): 
            sys.argv.remove('--build-js') 
         if ('--install-js' in sys.argv): 
            sys.argv.remove('--install-js') 
      if ('--existing-js' not in sys.argv): 
         sys.argv.append('--existing-js')

def _create_scheduled_actions(conn, as_name, scheduled_actions): 
    if scheduled_actions: 
      for (name, action) in six.iteritems(scheduled_actions): 
         if (('start_time' in action) and isinstance(action['start_time'], six.string_types)): 
            action['start_time'] = datetime.datetime.strptime(action['start_time'], DATE_FORMAT) 
         if (('end_time' in action) and isinstance(action['end_time'], six.string_types)): 
            action['end_time'] = datetime.datetime.strptime(action['end_time'], DATE_FORMAT) 
         conn.create_scheduled_group_action(as_name, name, desired_capacity=action.get('desired_capacity'), min_size=action.get('min_size'), max_size=action.get('max_size'), start_time=action.get('start_time'), end_time=action.get('end_time'), recurrence=action.get('recurrence'))

def _collect_delete_commands(base_mapper, uowtransaction, table, states_to_delete): 
    for (state, state_dict, mapper, connection, update_version_id) in states_to_delete: 
      if (table not in mapper._pks_by_table): 
         continue 
      params = {} 
      for col in mapper._pks_by_table[table]: 
         params[col.key] = value = mapper._get_committed_state_attr_by_column(state, state_dict, col) 
         if (value is None): 
            raise orm_exc.FlushError(("Can't   delete   from   table   %s   using   NULL   for   primary   key   value   on   column   %s" % (table, col))) 
      if ((update_version_id is not None) and (mapper.version_id_col in mapper._cols_by_table[table])): 
         params[mapper.version_id_col.key] = update_version_id 
      (yield (params, connection))

def eval_master_func(opts): 
    if ('__master_func_evaluated' not in opts): 
      mod_fun = opts['master'] 
      (mod, fun) = mod_fun.split('.') 
      try: 
         master_mod = salt.loader.raw_mod(opts, mod, fun) 
         if (not master_mod): 
            raise KeyError 
         opts['master'] = master_mod[mod_fun]() 
         if ((not isinstance(opts['master'], str)) and (not isinstance(opts['master'], list))): 
            raise TypeError 
         opts['__master_func_evaluated'] = True 
      except KeyError: 
         log.error('Failed   to   load   module   {0}'.format(mod_fun)) 
         sys.exit(salt.defaults.exitcodes.EX_GENERIC) 
      except TypeError: 
         log.error('{0}   returned   from   {1}   is   not   a   string   or   a   list'.format(opts['master'], mod_fun)) 
         sys.exit(salt.defaults.exitcodes.EX_GENERIC) 
      log.info('Evaluated   master   from   module:   {0}'.format(mod_fun))

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

def coerce_dtypes(df, dtypes): 
    for c in df.columns: 
      if ((c in dtypes) and (df.dtypes[c] != dtypes[c])): 
         if (np.issubdtype(df.dtypes[c], np.floating) and np.issubdtype(dtypes[c], np.integer)): 
            if (df[c] % 1).any(): 
               msg = "Runtime   type   mismatch.   Add   {'%s':   float}   to   dtype=   keyword   in   read_csv/read_table" 
               raise TypeError((msg % c)) 
         df[c] = df[c].astype(dtypes[c])

def process_rst_and_summaries(content_generators): 
    for generator in content_generators: 
      if isinstance(generator, generators.ArticlesGenerator): 
         for article in ((generator.articles + generator.translations) + generator.drafts): 
            rst_add_mathjax(article) 
            if (process_summary.mathjax_script is not None): 
               process_summary(article) 
      elif isinstance(generator, generators.PagesGenerator): 
         for page in generator.pages: 
            rst_add_mathjax(page)

def resolve_duplicates(session, task): 
    if (task.choice_flag in (action.ASIS, action.APPLY, action.RETAG)): 
      found_duplicates = task.find_duplicates(session.lib) 
      if found_duplicates: 
         log.debug(u'found   duplicates:   {}'.format([o.id for o in found_duplicates])) 
         duplicate_action = config['import']['duplicate_action'].as_choice({u'skip': u's', u'keep': u'k', u'remove': u'r', u'ask': u'a'}) 
         log.debug(u'default   action   for   duplicates:   {0}', duplicate_action) 
         if (duplicate_action == u's'): 
            task.set_choice(action.SKIP) 
         elif (duplicate_action == u'k'): 
            pass 
         elif (duplicate_action == u'r'): 
            task.should_remove_duplicates = True 
         else: 
            session.resolve_duplicate(task, found_duplicates) 
         session.log_choice(task, True)

def set_permissions(path, recursive=True): 
    if (not sabnzbd.WIN32): 
      umask = cfg.umask() 
      try: 
         umask = (int(umask, 8) | int('0700', 8)) 
         report = True 
      except ValueError: 
         umask = (int('0777', 8) & (sabnzbd.ORG_UMASK ^ int('0777', 8))) 
         report = False 
      umask_file = (umask & int('7666', 8)) 
      if os.path.isdir(path): 
         if recursive: 
            for (root, _dirs, files) in os.walk(path): 
               set_chmod(root, umask, report) 
               for name in files: 
                  set_chmod(os.path.join(root, name), umask_file, report) 
         else: 
            set_chmod(path, umask, report) 
      else: 
         set_chmod(path, umask_file, report)

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

def generate_alias(tbl): 
    return u''.join(([l for l in tbl if l.isupper()] or [l for (l, prev) in zip(tbl, (u'_' + tbl)) if ((prev == u'_') and (l != u'_'))]))

def makeRst(prefix, section, app, exampleByIdentifier, schema_store): 
    for route in sorted(getRoutes(app)): 
      if route.attributes.get('private_api', False): 
         continue 
      data = _introspectRoute(route, exampleByIdentifier, schema_store) 
      if (data['section'] != section): 
         continue 
      for method in route.methods: 
         if (data['header'] is not None): 
            (yield data['header']) 
            (yield ('-' * len(data['header']))) 
            (yield '') 
         body = _formatRouteBody(data, schema_store) 
         for line in http_directive(method, (prefix + route.path), body): 
            (yield line)

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

def eval_once(saver, summary_writer, top_k_op, summary_op): 
    with tf.Session() as sess: 
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir) 
      if (ckpt and ckpt.model_checkpoint_path): 
         saver.restore(sess, ckpt.model_checkpoint_path) 
         global_step = ckpt.model_checkpoint_path.split('/')[(-1)].split('-')[(-1)] 
      else: 
         print('No   checkpoint   file   found') 
         return 
      coord = tf.train.Coordinator() 
      try: 
         threads = [] 
         for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS): 
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True)) 
         num_iter = int(math.ceil((FLAGS.num_examples / FLAGS.batch_size))) 
         true_count = 0 
         total_sample_count = (num_iter * FLAGS.batch_size) 
         step = 0 
         while ((step < num_iter) and (not coord.should_stop())): 
            predictions = sess.run([top_k_op]) 
            true_count += np.sum(predictions) 
            step += 1 
         precision = (true_count / total_sample_count) 
         print(('%s:   precision   @   1   =   %.3f' % (datetime.now(), precision))) 
         summary = tf.Summary() 
         summary.ParseFromString(sess.run(summary_op)) 
         summary.value.add(tag='Precision   @   1', simple_value=precision) 
         summary_writer.add_summary(summary, global_step) 
      except Exception as e: 
         coord.request_stop(e) 
      coord.request_stop() 
      coord.join(threads, stop_grace_period_secs=10)

def pretty_all(container): 
    for (name, mt) in container.mime_map.iteritems(): 
      prettied = False 
      if (mt in OEB_DOCS): 
         pretty_html_tree(container, container.parsed(name)) 
         prettied = True 
      elif (mt in OEB_STYLES): 
         container.parsed(name) 
         prettied = True 
      elif (name == container.opf_name): 
         root = container.parsed(name) 
         pretty_opf(root) 
         pretty_xml_tree(root) 
         prettied = True 
      elif (mt in {guess_type(u'a.ncx'), guess_type(u'a.xml')}): 
         pretty_xml_tree(container.parsed(name)) 
         prettied = True 
      if prettied: 
         container.dirty(name)

def do_login(sender, user, request, **kwargs): 
    if (user and user.is_authenticated()): 
      token = None 
      try: 
         Application = get_application_model() 
         app = Application.objects.get(name='GeoServer') 
         token = generate_token() 
         AccessToken.objects.get_or_create(user=user, application=app, expires=(datetime.datetime.now() + datetime.timedelta(days=1)), token=token) 
      except: 
         u = uuid.uuid1() 
         token = u.hex 
      url = ('%s%s?access_token=%s' % (settings.OGC_SERVER['default']['PUBLIC_LOCATION'], 'ows?service=wms&version=1.3.0&request=GetCapabilities', token)) 
      cj = cookielib.CookieJar() 
      opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj)) 
      jsessionid = None 
      try: 
         opener.open(url) 
         for c in cj: 
            if (c.name == 'JSESSIONID'): 
               jsessionid = c.value 
      except: 
         u = uuid.uuid1() 
         jsessionid = u.hex 
      request.session['access_token'] = token 
      request.session['JSESSIONID'] = jsessionid

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

def eventloop(conn, limit=None, timeout=None, ignore_timeouts=False): 
    for i in ((limit and range(limit)) or count()): 
      try: 
         (yield conn.drain_events(timeout=timeout)) 
      except socket.timeout: 
         if (timeout and (not ignore_timeouts)): 
            raise

def resnet_v2(inputs, blocks, num_classes=None, is_training=True, global_pool=True, output_stride=None, include_root_block=True, reuse=None, scope=None): 
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc: 
      end_points_collection = (sc.name + '_end_points') 
      with slim.arg_scope([slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense], outputs_collections=end_points_collection): 
         with slim.arg_scope([slim.batch_norm], is_training=is_training): 
            net = inputs 
            if include_root_block: 
               if (output_stride is not None): 
                  if ((output_stride % 4) != 0): 
                     raise ValueError('The   output_stride   needs   to   be   a   multiple   of   4.') 
                  output_stride /= 4 
               with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None): 
                  net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1') 
               net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1') 
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride) 
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm') 
            if global_pool: 
               net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True) 
            if (num_classes is not None): 
               net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits') 
            end_points = slim.utils.convert_collection_to_dict(end_points_collection) 
            if (num_classes is not None): 
               end_points['predictions'] = slim.softmax(net, scope='predictions') 
            return (net, end_points)

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

def RemoveSelfDependencies(targets): 
    for (target_name, target_dict) in targets.iteritems(): 
      for dependency_key in dependency_sections: 
         dependencies = target_dict.get(dependency_key, []) 
         if dependencies: 
            for t in dependencies: 
               if (t == target_name): 
                  if targets[t].get('variables', {}).get('prune_self_dependency', 0): 
                     target_dict[dependency_key] = Filter(dependencies, target_name)

def iter_format_modules(lang): 
    if check_for_language(lang): 
      format_locations = ['django.conf.locale.%s'] 
      if settings.FORMAT_MODULE_PATH: 
         format_locations.append((settings.FORMAT_MODULE_PATH + '.%s')) 
         format_locations.reverse() 
      locale = to_locale(lang) 
      locales = [locale] 
      if ('_' in locale): 
         locales.append(locale.split('_')[0]) 
      for location in format_locations: 
         for loc in locales: 
            try: 
               (yield import_module(('%s.formats' % (location % loc)))) 
            except ImportError: 
               pass

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

def iter_format_modules(lang): 
    if check_for_language(lang): 
      format_locations = ['django.conf.locale.%s'] 
      if settings.FORMAT_MODULE_PATH: 
         format_locations.append((settings.FORMAT_MODULE_PATH + '.%s')) 
         format_locations.reverse() 
      locale = to_locale(lang) 
      locales = [locale] 
      if ('_' in locale): 
         locales.append(locale.split('_')[0]) 
      for location in format_locations: 
         for loc in locales: 
            try: 
               (yield import_module('.formats', (location % loc))) 
            except ImportError: 
               pass

def check_valid_file_exists(in_files): 
    for in_file in in_files: 
      if (in_file == u'-'): 
         pass 
      elif os.path.exists(in_file): 
         mode = os.stat(in_file).st_mode 
         if ((os.stat(in_file).st_size > 0) or S_ISBLK(mode) or S_ISFIFO(mode)): 
            return 
         else: 
            print((u'WARNING:   Input   file   %s   is   empty' % in_file), file=sys.stderr) 
      else: 
         print((u'WARNING:   Input   file   %s   not   found' % in_file), file=sys.stderr)

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

def _set_tcp_keepalive(zmq_socket, opts): 
    if (hasattr(zmq, 'TCP_KEEPALIVE') and opts): 
      if ('tcp_keepalive' in opts): 
         zmq_socket.setsockopt(zmq.TCP_KEEPALIVE, opts['tcp_keepalive']) 
      if ('tcp_keepalive_idle' in opts): 
         zmq_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, opts['tcp_keepalive_idle']) 
      if ('tcp_keepalive_cnt' in opts): 
         zmq_socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, opts['tcp_keepalive_cnt']) 
      if ('tcp_keepalive_intvl' in opts): 
         zmq_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, opts['tcp_keepalive_intvl'])

def mro_lookup(cls, attr, stop=set(), monkey_patched=[]): 
    for node in cls.mro(): 
      if (node in stop): 
         try: 
            value = node.__dict__[attr] 
            module_origin = value.__module__ 
         except (AttributeError, KeyError): 
            pass 
         else: 
            if (module_origin not in monkey_patched): 
               return node 
         return 
      if (attr in node.__dict__): 
         return node

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

def _recurse_config_to_dict(t_data): 
    if (not isinstance(t_data, type(None))): 
      if isinstance(t_data, list): 
         t_list = [] 
         for i in t_data: 
            t_list.append(_recurse_config_to_dict(i)) 
         return t_list 
      elif isinstance(t_data, dict): 
         t_dict = {} 
         for (k, v) in t_data.iteritems(): 
            t_dict[k] = _recurse_config_to_dict(v) 
         return t_dict 
      elif hasattr(t_data, '__dict__'): 
         return _recurse_config_to_dict(t_data.__dict__) 
      else: 
         return _serializer(t_data)

