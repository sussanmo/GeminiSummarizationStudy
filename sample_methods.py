def sign_int(message, dkey, n): 
    return decrypt_int(message, dkey, n)

def get_colormaps(): 
    return _colormaps.copy()

def _ems_diff(data0, data1): 
    return (np.mean(data0, axis=0) - np.mean(data1, axis=0))

def normalize(x, eps=1e-05): 
    return NormalizeL2(eps)(x)

def randrange(n, vmin, vmax): 
    return (((vmax - vmin) * np.random.rand(n)) + vmin)

def split_and_strip_non_empty_lines(text): 
    return [line.strip() for line in text.splitlines() if line.strip()]

def get_signed_purchase_params(cart, callback_url=None, extra_data=None): 
    return sign(get_purchase_params(cart, callback_url=callback_url, extra_data=extra_data))

def ngettext(singular, plural, number): 
    return do_ntranslate(singular, plural, number, 'ngettext')

def _get_xtool(): 
    for xtool in ['xl', 'xm']: 
      path = salt.utils.which(xtool) 
      if (path is not None): 
         return path

def _write_file_network(data, filename): 
    with salt.utils.fopen(filename, 'w') as fp_: 
      fp_.write(data)

def isValid(text): 
    return bool(re.search('\\bemail\\b', text, re.IGNORECASE))

def _normalize_proj(info): 
    _make_projector(info['projs'], info.get('ch_names', info.get('names')), info['bads'], include_active=True, inplace=True)

def __virtual__(): 
    return ('selinux' if ('selinux.getenforce' in __salt__) else False)

def cgsnapshot_creating_from_src(): 
    return sql.exists().where(and_((models.Cgsnapshot.consistencygroup_id == models.ConsistencyGroup.id), (~ models.Cgsnapshot.deleted), (models.Cgsnapshot.status == 'creating')))

def clear_search_index(): 
    search_services.clear_index(SEARCH_INDEX_COLLECTIONS)

def cloudstack_displayname(vm_): 
    return config.get_cloud_config_value('cloudstack_displayname', vm_, __opts__, search_global=True)

def random_bitstring(n): 
    return ''.join([random.choice('01') for i in range(n)])

def in6_isuladdr(str): 
    return in6_isincluded(str, 'fc00::', 7)

def set_default_zone(zone): 
    return __firewall_cmd('--set-default-zone={0}'.format(zone))

def literal_string(s): 
    return ((u"'" + s.replace(u"'", u"''").replace(u'\x00', '')) + u"'")

def comment(path, regex, char='#', backup='.bak'): 
    return comment_line(path=path, regex=regex, char=char, cmnt=True, backup=backup)

def setup_py_test(): 
    nose.main(addplugins=[NoseSQLAlchemy()], argv=['runner'])

def __virtual__(): 
    return ('pagerduty_user' if ('pagerduty_util.get_resource' in __salt__) else False)

def track_for_id(track_id): 
    for plugin in find_plugins(): 
      track = plugin.track_for_id(track_id) 
      if track: 
         (yield track)

def get_widgets(request): 
    return WIDGETS

def after_scenario(context, _): 
    if (hasattr(context, u'cli') and (not context.exit_sent)): 
      context.cli.terminate()

def migration_get_unconfirmed_by_dest_compute(context, confirm_window, dest_compute): 
    return IMPL.migration_get_unconfirmed_by_dest_compute(context, confirm_window, dest_compute)

def _chop(seq, how_much): 
    return seq[_B(how_much):]

def _sort_candidates(candidates): 
    return sorted(candidates, key=(lambda match: match.distance))

def get_user(bootinfo): 
    bootinfo.user = frappe.get_user().load_user()

def tac(): 
    return __timer__.tac()

def framework(): 
    return s3_rest_controller(dtargs={'dt_text_maximum_len': 160}, hide_filter=True)

def _patch_object(target, attribute, new=DEFAULT, spec=None, create=False, spec_set=None, autospec=None, new_callable=None, **kwargs): 
    return _patch((lambda : target), attribute, new, spec, create, spec_set, autospec, new_callable, kwargs)

def dictsortreversed(value, arg): 
    return sorted(value, key=Variable(arg).resolve, reverse=True)

def softmax(x): 
    return tf.nn.softmax(x)

def expandvars(path): 
    return path

def validate_mapped_group_ids(group_ids, mapping_id, identity_api): 
    for group_id in group_ids: 
      try: 
         identity_api.get_group(group_id) 
      except exception.GroupNotFound: 
         raise exception.MappedGroupNotFound(group_id=group_id, mapping_id=mapping_id)

def refresh_db(): 
    return (__salt__['cmd.retcode']('/opt/csw/bin/pkgutil   -U') == 0)

def get_bootstrap_setting(setting, default=None): 
    return BOOTSTRAP3.get(setting, default)

def CPP_INTEGER(t): 
    return t

def get_interface_version(): 
    return INTERFACE_VERSION

def test_boolean(): 
    _test_interop_set(clr_types, py_types, bool_test_cases)

def setup_logging(loglevel=logging.DEBUG, loggers=[u'kombu.connection', u'kombu.channel']): 
    for logger in loggers: 
      l = get_logger(logger) 
      l.addHandler(logging.StreamHandler()) 
      l.setLevel(loglevel)

def rospack_depends_on(pkg): 
    return rospackexec(['depends-on', pkg]).split()

def _api_undefined(name, output, kwargs): 
    return report(output, _MSG_NOT_IMPLEMENTED)

def get_other_props(all_props, reserved_props): 
    if (hasattr(all_props, 'items') and callable(all_props.items)): 
      return dict([(k, v) for (k, v) in all_props.items() if (k not in reserved_props)])

def shutdown(handlerList=_handlerList): 
    for wr in reversed(handlerList[:]): 
      try: 
         h = wr() 
         if h: 
            try: 
               h.acquire() 
               h.flush() 
               h.close() 
            except (IOError, ValueError): 
               pass 
            finally: 
               h.release() 
      except: 
         if raiseExceptions: 
            raise

def input_loop(): 
    while (mpstate.status.exit != True): 
      try: 
         if (mpstate.status.exit != True): 
            line = raw_input(mpstate.rl.prompt) 
      except EOFError: 
         mpstate.status.exit = True 
         sys.exit(1) 
      mpstate.input_queue.put(line)

def rec_test(sequence, test_func): 
    for x in sequence: 
      if isinstance(x, (list, tuple)): 
         for y in rec_test(x, test_func): 
            (yield y) 
      else: 
         (yield test_func(x))

def dn2str(dn): 
    return ','.join(['+'.join(['='.join((atype, escape_dn_chars((avalue or '')))) for (atype, avalue, dummy) in rdn]) for rdn in dn])

def next_char(input_iter): 
    for ch in input_iter: 
      if (ch != u'\\'): 
         (yield (ch, False)) 
         continue 
      ch = next(input_iter) 
      representative = ESCAPE_MAPPINGS.get(ch, ch) 
      if (representative is None): 
         continue 
      (yield (representative, True))

def _deserialize_dependencies(artifact_type, deps_from_db, artifact_properties, plugins): 
    for (dep_name, dep_value) in six.iteritems(deps_from_db): 
      if (not dep_value): 
         continue 
      if isinstance(artifact_type.metadata.attributes.dependencies.get(dep_name), declarative.ListAttributeDefinition): 
         val = [] 
         for v in dep_value: 
            val.append(deserialize_from_db(v, plugins)) 
      elif (len(dep_value) == 1): 
         val = deserialize_from_db(dep_value[0], plugins) 
      else: 
         raise exception.InvalidArtifactPropertyValue(message=_('Relation   %(name)s   may   not   have   multiple   values'), name=dep_name) 
      artifact_properties[dep_name] = val

def _list_files(folder, pattern): 
    for (root, folders, files) in os.walk(folder): 
      for filename in files: 
         if fnmatch.fnmatch(filename, pattern): 
            (yield os.path.join(root, filename))

def _add_implied_job_id(d): 
    if (not d.get('job_id')): 
      if d.get('task_id'): 
         d['job_id'] = _to_job_id(d['task_id']) 
      elif d.get('application_id'): 
         d['job_id'] = _to_job_id(d['application_id'])

def warn_exception(exc, **kargs): 
    return ('WARNING:   %s   [%r]%s\n%s' % (exc, exc, (('   [%s]' % ',   '.join((('%s=%s' % (key, value)) for (key, value) in kargs.iteritems()))) if kargs else ''), ((' DCTB %s\n' % '\n DCTB '.join(traceback.format_exc().splitlines())) if config.DEBUG else '')))

def iter_platform_files(dst): 
    for (root, dirs, files) in os.walk(dst): 
      for fn in files: 
         fn = os.path.join(root, fn) 
         if is_platform_file(fn): 
            (yield fn)

def query_package(module, pacman_path, name, state='present'): 
    if (state == 'present'): 
      lcmd = ('%s   -Qi   %s' % (pacman_path, name)) 
      (lrc, lstdout, lstderr) = module.run_command(lcmd, check_rc=False) 
      if (lrc != 0): 
         return (False, False, False) 
      lversion = get_version(lstdout) 
      rcmd = ('%s   -Si   %s' % (pacman_path, name)) 
      (rrc, rstdout, rstderr) = module.run_command(rcmd, check_rc=False) 
      rversion = get_version(rstdout) 
      if (rrc == 0): 
         return (True, (lversion == rversion), False) 
      return (True, True, True)

def _validate_numa_nodes(nodes): 
    if ((nodes is not None) and ((not strutils.is_int_like(nodes)) or (int(nodes) < 1))): 
      raise exception.InvalidNUMANodesNumber(nodes=nodes)

def find_selected(nodes): 
    for node in nodes: 
      if hasattr(node, 'selected'): 
         return node 
      elif hasattr(node, 'ancestor'): 
         result = find_selected(node.children) 
         if result: 
            return result

def col(loc, strg): 
    return ((((loc < len(strg)) and (strg[loc] == '\n')) and 1) or (loc - strg.rfind('\n', 0, loc)))

def chown(path, owner=None): 
    if owner: 
      try: 
         (x, y) = (owner, (-1)) 
         (x, y) = (x if isinstance(x, tuple) else (x, y)) 
         (x, y) = ((pwd.getpwnam(x).pw_uid if (not isinstance(x, int)) else x), (grp.getgrnam(y).gr_gid if (not isinstance(y, int)) else y)) 
         os.chown(path, x, y) 
         return True 
      except: 
         return False

def _stop_timers(canvas): 
    for attr in dir(canvas): 
      try: 
         attr_obj = getattr(canvas, attr) 
      except NotImplementedError: 
         attr_obj = None 
      if isinstance(attr_obj, Timer): 
         attr_obj.stop()

def process_contexts(server, contexts, p_ctx, error=None): 
    for ctx in contexts: 
      ctx.descriptor.aux.initialize_context(ctx, p_ctx, error) 
      if ((error is None) or ctx.descriptor.aux.process_exceptions): 
         ctx.descriptor.aux.process_context(server, ctx)

def check_freezing_date(posting_date, adv_adj=False): 
    if (not adv_adj): 
      acc_frozen_upto = frappe.db.get_value(u'Accounts   Settings', None, u'acc_frozen_upto') 
      if acc_frozen_upto: 
         frozen_accounts_modifier = frappe.db.get_value(u'Accounts   Settings', None, u'frozen_accounts_modifier') 
         if ((getdate(posting_date) <= getdate(acc_frozen_upto)) and (not (frozen_accounts_modifier in frappe.get_roles()))): 
            frappe.throw(_(u'You   are   not   authorized   to   add   or   update   entries   before   {0}').format(formatdate(acc_frozen_upto)))

def base_search(index, query, params, search, schema): 
    with index.searcher() as searcher: 
      queries = [] 
      for param in params: 
         if search[param]: 
            parser = qparser.QueryParser(param, schema) 
            queries.append(parser.parse(query)) 
      terms = functools.reduce((lambda x, y: (x | y)), queries) 
      return [result['pk'] for result in searcher.search(terms)]

def wsgi_xmlrpc(environ, start_response): 
    if ((environ['REQUEST_METHOD'] == 'POST') and environ['PATH_INFO'].startswith('/xmlrpc/')): 
      length = int(environ['CONTENT_LENGTH']) 
      data = environ['wsgi.input'].read(length) 
      string_faultcode = True 
      if environ['PATH_INFO'].startswith('/xmlrpc/2/'): 
         service = environ['PATH_INFO'][len('/xmlrpc/2/'):] 
         string_faultcode = False 
      else: 
         service = environ['PATH_INFO'][len('/xmlrpc/'):] 
      (params, method) = xmlrpclib.loads(data) 
      return xmlrpc_return(start_response, service, method, params, string_faultcode)

def split_named_range(range_string): 
    for range_string in SPLIT_NAMED_RANGE_RE.split(range_string)[1::2]: 
      match = NAMED_RANGE_RE.match(range_string) 
      if (match is None): 
         raise NamedRangeException(('Invalid   named   range   string:   "%s"' % range_string)) 
      else: 
         match = match.groupdict() 
         sheet_name = (match['quoted'] or match['notquoted']) 
         xlrange = match['range'] 
         sheet_name = sheet_name.replace("''", "'") 
         (yield (sheet_name, xlrange))

def each_setup_in_pkg(top_dir): 
    for (dir_path, dir_names, filenames) in os.walk(top_dir): 
      for fname in filenames: 
         if (fname == 'setup.py'): 
            with open(os.path.join(dir_path, 'setup.py')) as f: 
               (yield (dir_path, f))

def previous_key(tuple_of_tuples, key): 
    for (i, t) in enumerate(tuple_of_tuples): 
      if (t[0] == key): 
         try: 
            return tuple_of_tuples[(i - 1)][0] 
         except IndexError: 
            return None

def key_not_string(d): 
    for (k, v) in d.items(): 
      if ((not isinstance(k, six.string_types)) or (isinstance(v, dict) and key_not_string(v))): 
         return True

def service_tags_not_in_module_path(physical_line, filename): 
    if ('tempest/scenario' not in filename): 
      matches = SCENARIO_DECORATOR.match(physical_line) 
      if matches: 
         services = matches.group(1).split(',') 
         for service in services: 
            service_name = service.strip().strip("'") 
            modulepath = os.path.split(filename)[0] 
            if (service_name in modulepath): 
               return (physical_line.find(service_name), 'T107:   service   tag   should   not   be   in   path')

def parse_token_stream(stream, soft_delimiter, hard_delimiter): 
    return [[sum((len(token) for token in sentence_it)) for sentence_it in split_at(block_it, soft_delimiter)] for block_it in split_at(stream, hard_delimiter)]

def readAuthorizedKeyFile(fileobj, parseKey=keys.Key.fromString): 
    for line in fileobj: 
      line = line.strip() 
      if (line and (not line.startswith('#'))): 
         try: 
            (yield parseKey(line)) 
         except keys.BadKeyError as e: 
            log.msg('Unable   to   parse   line   "{0}"   as   a   key:   {1!s}'.format(line, e))

def wait(service, condition, fail_condition=(lambda e: False), timeout=180, wait=True, poll_interval=3): 
    if wait: 
      start = time.time() 
      while (time.time() < (start + timeout)): 
         entity = get_entity(service) 
         if condition(entity): 
            return 
         elif fail_condition(entity): 
            raise Exception('Error   while   waiting   on   result   state   of   the   entity.') 
         time.sleep(float(poll_interval))

def _find_match(ele, lst): 
    for _ele in lst: 
      for match_key in _MATCH_KEYS: 
         if (_ele.get(match_key) == ele.get(match_key)): 
            return _ele

def list_themes(v=False): 
    for (t, l) in themes(): 
      if (not v): 
         t = os.path.basename(t) 
      if l: 
         if v: 
            print((t + ((u'   (symbolic   link   to   `' + l) + u"')"))) 
         else: 
            print((t + u'@')) 
      else: 
         print(t)

def is_attr_protected(attrname): 
    return ((attrname[0] == '_') and (not (attrname == '_')) and (not (attrname.startswith('__') and attrname.endswith('__'))))

def no_vi_headers(physical_line, line_number, lines): 
    if ((line_number <= 5) or (line_number > (len(lines) - 5))): 
      if VI_HEADER_RE.match(physical_line): 
         return (0, "T106:   Don't   put   vi   configuration   in   source   files")

def get_chunks_in_range(chunks, first_line, num_lines): 
    for (i, chunk) in enumerate(chunks): 
      lines = chunk[u'lines'] 
      if (lines[(-1)][0] >= first_line >= lines[0][0]): 
         start_index = (first_line - lines[0][0]) 
         if ((first_line + num_lines) <= lines[(-1)][0]): 
            last_index = (start_index + num_lines) 
         else: 
            last_index = len(lines) 
         new_chunk = {u'index': i, u'lines': chunk[u'lines'][start_index:last_index], u'numlines': (last_index - start_index), u'change': chunk[u'change'], u'meta': chunk.get(u'meta', {})} 
         (yield new_chunk) 
         first_line += new_chunk[u'numlines'] 
         num_lines -= new_chunk[u'numlines'] 
         assert (num_lines >= 0) 
         if (num_lines == 0): 
            break

def get_numpy_dtype(obj): 
    if (ndarray is not FakeObject): 
      import numpy as np 
      if (isinstance(obj, np.generic) or isinstance(obj, np.ndarray)): 
         try: 
            return obj.dtype.type 
         except (AttributeError, RuntimeError): 
            return

def resolve_ambiguity(all_tokens, seen_ts): 
    for (parent, token) in all_tokens: 
      if isinstance(token, MirrorToken): 
         if (token.number not in seen_ts): 
            seen_ts[token.number] = TabStop(parent, token) 
         else: 
            Mirror(parent, seen_ts[token.number], token)

def ask(message, options): 
    while 1: 
      if os.environ.get('PIP_NO_INPUT'): 
         raise Exception(('No   input   was   expected   ($PIP_NO_INPUT   set);   question:   %s' % message)) 
      response = raw_input(message) 
      response = response.strip().lower() 
      if (response not in options): 
         print ('Your   response   (%r)   was   not   one   of   the   expected   responses:   %s' % (response, ',   '.join(options))) 
      else: 
         return response

def get_metadata(headers): 
    return dict(((k, v) for (k, v) in headers.iteritems() if any((k.lower().startswith(valid) for valid in _GCS_METADATA))))

def first(seq, key=(lambda x: bool(x)), default=None, apply=(lambda x: x)): 
    return next((apply(x) for x in seq if key(x)), (default() if callable(default) else default))

def check_abstract_methods(base, subclass): 
    for attrname in dir(base): 
      if attrname.startswith('_'): 
         continue 
      attr = getattr(base, attrname) 
      if is_abstract_method(attr): 
         oattr = getattr(subclass, attrname) 
         if is_abstract_method(oattr): 
            raise Exception(('%s.%s   not   overridden' % (subclass.__name__, attrname)))

def _writeFlattenedData(state, write, result): 
    while True: 
      try: 
         element = next(state) 
      except StopIteration: 
         result.callback(None) 
      except: 
         result.errback() 
      else: 
         def cby(original): 
            _writeFlattenedData(state, write, result) 
            return original 
         element.addCallbacks(cby, result.errback) 
      break

def addXIntersectionIndexesFromLoop(frontOverWidth, loop, solidIndex, xIntersectionIndexLists, width, yList): 
    for pointIndex in xrange(len(loop)): 
      pointBegin = loop[pointIndex] 
      pointEnd = loop[((pointIndex + 1) % len(loop))] 
      if (pointBegin.imag > pointEnd.imag): 
         pointOriginal = pointBegin 
         pointBegin = pointEnd 
         pointEnd = pointOriginal 
      fillBegin = int(math.ceil(((pointBegin.imag / width) - frontOverWidth))) 
      fillBegin = max(0, fillBegin) 
      fillEnd = int(math.ceil(((pointEnd.imag / width) - frontOverWidth))) 
      fillEnd = min(len(xIntersectionIndexLists), fillEnd) 
      if (fillEnd > fillBegin): 
         secondMinusFirstComplex = (pointEnd - pointBegin) 
         secondMinusFirstImaginaryOverReal = (secondMinusFirstComplex.real / secondMinusFirstComplex.imag) 
         beginRealMinusImaginary = (pointBegin.real - (pointBegin.imag * secondMinusFirstImaginaryOverReal)) 
         for fillLine in xrange(fillBegin, fillEnd): 
            xIntersection = ((yList[fillLine] * secondMinusFirstImaginaryOverReal) + beginRealMinusImaginary) 
            xIntersectionIndexList = xIntersectionIndexLists[fillLine] 
            xIntersectionIndexList.append(XIntersectionIndex(solidIndex, xIntersection))

def _gpa11iterator(handle): 
    for inline in handle: 
      if (inline[0] == '!'): 
         continue 
      inrec = inline.rstrip('\n').split(' DCTB ') 
      if (len(inrec) == 1): 
         continue 
      inrec[2] = inrec[2].split('|') 
      inrec[4] = inrec[4].split('|') 
      inrec[6] = inrec[6].split('|') 
      inrec[10] = inrec[10].split('|') 
      (yield dict(zip(GPA11FIELDS, inrec)))

def create_fakedir(outputdir, tiles): 
    for (tilepath, tilemtime) in tiles.iteritems(): 
      dirpath = os.path.join(outputdir, *(str(x) for x in tilepath[:(-1)])) 
      if (len(tilepath) == 0): 
         imgname = 'base.png' 
      else: 
         imgname = (str(tilepath[(-1)]) + '.png') 
      if (not os.path.exists(dirpath)): 
         os.makedirs(dirpath) 
      finalpath = os.path.join(dirpath, imgname) 
      open(finalpath, 'w').close() 
      os.utime(finalpath, (tilemtime, tilemtime))

def get_tree_changes(repo): 
    with open_repo_closing(repo) as r: 
      index = r.open_index() 
      tracked_changes = {'add': [], 'delete': [], 'modify': []} 
      try: 
         tree_id = r['HEAD'].tree 
      except KeyError: 
         tree_id = None 
      for change in index.changes_from_tree(r.object_store, tree_id): 
         if (not change[0][0]): 
            tracked_changes['add'].append(change[0][1]) 
         elif (not change[0][1]): 
            tracked_changes['delete'].append(change[0][0]) 
         elif (change[0][0] == change[0][1]): 
            tracked_changes['modify'].append(change[0][0]) 
         else: 
            raise AssertionError('git   mv   ops   not   yet   supported') 
      return tracked_changes

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

def _is_import_valid(documents): 
    return (isinstance(documents, list) and all((isinstance(d, dict) for d in documents)) and all((all(((k in d) for k in ('pk', 'model', 'fields'))) for d in documents)) and all((all(((k in d['fields']) for k in ('uuid', 'owner'))) for d in documents)))

def write_worksheet_cols(doc, worksheet): 
    if worksheet.column_dimensions: 
      start_tag(doc, 'cols') 
      for (column_string, columndimension) in worksheet.column_dimensions.items(): 
         col_index = column_index_from_string(column_string) 
         col_def = {} 
         col_def['collapsed'] = str(columndimension.style_index) 
         col_def['min'] = str(col_index) 
         col_def['max'] = str(col_index) 
         if (columndimension.width != worksheet.default_column_dimension.width): 
            col_def['customWidth'] = 'true' 
         if (not columndimension.visible): 
            col_def['hidden'] = 'true' 
         if (columndimension.outline_level > 0): 
            col_def['outlineLevel'] = str(columndimension.outline_level) 
         if columndimension.collapsed: 
            col_def['collapsed'] = 'true' 
         if columndimension.auto_size: 
            col_def['bestFit'] = 'true' 
         if (columndimension.width > 0): 
            col_def['width'] = str(columndimension.width) 
         else: 
            col_def['width'] = '9.10' 
         tag(doc, 'col', col_def) 
      end_tag(doc, 'cols')

def _collapse_addresses_recursive(addresses): 
    while True: 
      last_addr = None 
      ret_array = [] 
      optimized = False 
      for cur_addr in addresses: 
         if (not ret_array): 
            last_addr = cur_addr 
            ret_array.append(cur_addr) 
         elif ((cur_addr.network_address >= last_addr.network_address) and (cur_addr.broadcast_address <= last_addr.broadcast_address)): 
            optimized = True 
         elif (cur_addr == list(last_addr.supernet().subnets())[1]): 
            ret_array[(-1)] = last_addr = last_addr.supernet() 
            optimized = True 
         else: 
            last_addr = cur_addr 
            ret_array.append(cur_addr) 
      addresses = ret_array 
      if (not optimized): 
         return addresses

def patch_crypto_be_discovery(): 
    if (((sys.platform == 'win32') or (sys.platform == 'darwin')) and is_frozen()): 
      from cryptography.hazmat import backends 
      try: 
         from cryptography.hazmat.backends.commoncrypto.backend import backend as be_cc 
      except ImportError: 
         be_cc = None 
      try: 
         from cryptography.hazmat.backends.openssl.backend import backend as be_ossl 
      except ImportError: 
         be_ossl = None 
      backends._available_backends_list = [be for be in (be_cc, be_ossl) if (be is not None)]

def process_failed(dirName, nzbName, result): 
    if sickrage.srCore.srConfig.USE_FAILED_DOWNLOADS: 
      processor = None 
      try: 
         processor = failed_processor.FailedProcessor(dirName, nzbName) 
         result.result = processor.process() 
         process_fail_message = u'' 
      except FailedPostProcessingFailedException as e: 
         result.result = False 
         process_fail_message = e 
      if processor: 
         result.output += processor.log 
      if (sickrage.srCore.srConfig.DELETE_FAILED and result.result): 
         if delete_folder(dirName, check_empty=False): 
            result.output += logHelper((u'Deleted   folder:   ' + dirName), sickrage.srCore.srLogger.DEBUG) 
      if result.result: 
         result.output += logHelper(((((u'Failed   Download   Processing   succeeded:   (' + str(nzbName)) + u',   ') + dirName) + u')')) 
      else: 
         result.output += logHelper(u'Failed   Download   Processing   failed:   ({},   {}):   {}'.format(nzbName, dirName, process_fail_message), sickrage.srCore.srLogger.WARNING)

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

def notify_unreplied(): 
    for email_account in frappe.get_all(u'Email   Account', u'name', filters={u'enable_incoming': 1, u'notify_if_unreplied': 1}): 
      email_account = frappe.get_doc(u'Email   Account', email_account.name) 
      if email_account.append_to: 
         for comm in frappe.get_all(u'Communication', u'name', filters={u'sent_or_received': u'Received', u'reference_doctype': email_account.append_to, u'unread_notification_sent': 0, u'email_account': email_account.name, u'creation': (u'<', (datetime.now() - timedelta(seconds=((email_account.unreplied_for_mins or 30) * 60)))), u'creation': (u'>', (datetime.now() - timedelta(seconds=(((email_account.unreplied_for_mins or 30) * 60) * 3))))}): 
            comm = frappe.get_doc(u'Communication', comm.name) 
            if (frappe.db.get_value(comm.reference_doctype, comm.reference_name, u'status') == u'Open'): 
               frappe.sendmail(recipients=email_account.get_unreplied_notification_emails(), content=comm.content, subject=comm.subject, doctype=comm.reference_doctype, name=comm.reference_name) 
            comm.db_set(u'unread_notification_sent', 1)

def test_client_options(config): 
    if config['use_ssl']: 
      if (('certificate' in config) and config['certificate']): 
         read_file(config['certificate']) 
      if (('client_cert' in config) and config['client_cert']): 
         read_file(config['client_cert']) 
      if (('client_key' in config) and config['client_key']): 
         read_file(config['client_key'])

def _is_def_line(line): 
    return (line.endswith(':') and (not ('class' in line.split())) and (line.startswith('def   ') or line.startswith('cdef   ') or line.startswith('cpdef   ') or ('   def   ' in line) or ('   cdef   ' in line) or ('   cpdef   ' in line)))

def _check_update_montage(info, montage, path=None, update_ch_names=False): 
    if (montage is not None): 
      if (not isinstance(montage, (string_types, Montage))): 
         err = ('Montage   must   be   str,   None,   or   instance   of   Montage.   %s   was   provided' % type(montage)) 
         raise TypeError(err) 
      if (montage is not None): 
         if isinstance(montage, string_types): 
            montage = read_montage(montage, path=path) 
         _set_montage(info, montage, update_ch_names=update_ch_names) 
         missing_positions = [] 
         exclude = (FIFF.FIFFV_EOG_CH, FIFF.FIFFV_MISC_CH, FIFF.FIFFV_STIM_CH) 
         for ch in info['chs']: 
            if (not (ch['kind'] in exclude)): 
               if (np.unique(ch['loc']).size == 1): 
                  missing_positions.append(ch['ch_name']) 
         if missing_positions: 
            raise KeyError(('The   following   positions   are   missing   from   the   montage   definitions:   %s.   If   those   channels   lack   positions   because   they   are   EOG   channels   use   the   eog   parameter.' % str(missing_positions)))

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

def recursive_rm(*patterns): 
    for (root, subdirs, subfiles) in os.walk('.'): 
      root = os.path.normpath(root) 
      if root.startswith('.git/'): 
         continue 
      for file in subfiles: 
         for pattern in patterns: 
            if fnmatch.fnmatch(file, pattern): 
               safe_remove(os.path.join(root, file)) 
      for dir in subdirs: 
         for pattern in patterns: 
            if fnmatch.fnmatch(dir, pattern): 
               safe_rmtree(os.path.join(root, dir))

def match(condition, data): 
    return ((condition == data) or (isinstance(condition, type) and isinstance(data, condition)) or ((not isinstance(condition, type)) and callable(condition) and condition(data)) or (isinstance(condition, tuple) and any((match(c, data) for c in condition))))

def ensure_cwltool_available(): 
    if ((main is None) or (workflow is None) or (shellescape is None)): 
      message = 'This   feature   requires   cwltool   and   dependencies   to   be   available,   they   are   not.' 
      if (main is None): 
         message += '   cwltool   is   not   unavailable.' 
      elif (load_tool is None): 
         message += '   cwltool.load_tool   is   unavailable   -   cwltool   version   is   too   old.' 
      if (requests is None): 
         message += "   Library   'requests'   unavailable." 
      if (shellescape is None): 
         message += "   Library   'shellescape'   unavailable." 
      if (schema_salad is None): 
         message += "   Library   'schema_salad'   unavailable." 
      raise ImportError(message)

def _generate_course_structure(course_key): 
    with modulestore().bulk_operations(course_key): 
      course = modulestore().get_course(course_key, depth=None) 
      blocks_stack = [course] 
      blocks_dict = {} 
      discussions = {} 
      while blocks_stack: 
         curr_block = blocks_stack.pop() 
         children = (curr_block.get_children() if curr_block.has_children else []) 
         key = unicode(curr_block.scope_ids.usage_id) 
         block = {'usage_key': key, 'block_type': curr_block.category, 'display_name': curr_block.display_name, 'children': [unicode(child.scope_ids.usage_id) for child in children]} 
         if ((curr_block.category == 'discussion') and hasattr(curr_block, 'discussion_id') and curr_block.discussion_id): 
            discussions[curr_block.discussion_id] = unicode(curr_block.scope_ids.usage_id) 
         attrs = (('graded', False), ('format', None)) 
         for (attr, default) in attrs: 
            if hasattr(curr_block, attr): 
               block[attr] = getattr(curr_block, attr, default) 
            else: 
               log.warning('Failed   to   retrieve   %s   attribute   of   block   %s.   Defaulting   to   %s.', attr, key, default) 
               block[attr] = default 
         blocks_dict[key] = block 
         blocks_stack.extend(children) 
      return {'structure': {'root': unicode(course.scope_ids.usage_id), 'blocks': blocks_dict}, 'discussion_id_map': discussions}

def find_triples(tokens, left_dependency_label='NSUBJ', head_part_of_speech='VERB', right_dependency_label='DOBJ'): 
    for (head, token) in enumerate(tokens): 
      if (token['partOfSpeech']['tag'] == head_part_of_speech): 
         children = dependents(tokens, head) 
         left_deps = [] 
         right_deps = [] 
         for child in children: 
            child_token = tokens[child] 
            child_dep_label = child_token['dependencyEdge']['label'] 
            if (child_dep_label == left_dependency_label): 
               left_deps.append(child) 
            elif (child_dep_label == right_dependency_label): 
               right_deps.append(child) 
         for left_dep in left_deps: 
            for right_dep in right_deps: 
               (yield (left_dep, head, right_dep))

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

def find_deprecated_defs(pkg_dir): 
    for (root, dirs, files) in os.walk(pkg_dir): 
      for filename in files: 
         if filename.endswith('.py'): 
            s = open(os.path.join(root, filename)).read() 
            for m in DEPRECATED_DEF_RE.finditer(s): 
               if m.group(2): 
                  name = m.group(2) 
                  msg = '   '.join((strip_quotes(s) for s in STRING_RE.findall(m.group(1)))) 
                  msg = '   '.join(msg.split()) 
                  if (m.group()[0] in '    DCTB '): 
                     cls = find_class(s, m.start()) 
                     deprecated_methods[name].add((msg, cls, '()')) 
                  else: 
                     deprecated_funcs[name].add((msg, '', '()')) 
               else: 
                  name = m.group(3) 
                  m2 = STRING_RE.match(s, m.end()) 
                  if m2: 
                     msg = strip_quotes(m2.group()) 
                  else: 
                     msg = '' 
                  msg = '   '.join(msg.split()) 
                  deprecated_classes[name].add((msg, '', ''))

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

def trade(events, strategy, portfolio, execution, heartbeat): 
    while True: 
      try: 
         event = events.get(False) 
      except queue.Empty: 
         pass 
      else: 
         if (event is not None): 
            if (event.type == 'TICK'): 
               logger.info('Received   new   tick   event:   %s', event) 
               strategy.calculate_signals(event) 
               portfolio.update_portfolio(event) 
            elif (event.type == 'SIGNAL'): 
               logger.info('Received   new   signal   event:   %s', event) 
               portfolio.execute_signal(event) 
            elif (event.type == 'ORDER'): 
               logger.info('Received   new   order   event:   %s', event) 
               execution.execute_order(event) 
      time.sleep(heartbeat)

def recursive_dict_removal(inventory, purge_list): 
    for (key, value) in inventory.iteritems(): 
      if isinstance(value, dict): 
         for (child_key, child_value) in value.iteritems(): 
            if isinstance(child_value, dict): 
               for item in purge_list: 
                  if (item in child_value): 
                     del child_value[item] 
            elif isinstance(child_value, list): 
               recursive_list_removal(child_value, purge_list) 
      elif isinstance(value, list): 
         recursive_list_removal(value, purge_list)

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

def _traverse_results(value, fields, row, path): 
    for (f, v) in value.iteritems(): 
      field_name = ('{path}.{name}'.format(path=path, name=f) if path else f) 
      if (not isinstance(v, (dict, list, tuple))): 
         if (field_name in fields): 
            row[fields.index(field_name)] = ensure_utf(v) 
      elif (isinstance(v, dict) and (f != 'attributes')): 
         _traverse_results(v, fields, row, field_name)

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

def AceIterator(handle): 
    for ace_contig in Ace.parse(handle): 
      consensus_seq_str = ace_contig.sequence 
      if ('U' in consensus_seq_str): 
         if ('T' in consensus_seq_str): 
            alpha = generic_nucleotide 
         else: 
            alpha = generic_rna 
      else: 
         alpha = generic_dna 
      if ('*' in consensus_seq_str): 
         assert ('-' not in consensus_seq_str) 
         consensus_seq = Seq(consensus_seq_str.replace('*', '-'), Gapped(alpha, gap_char='-')) 
      else: 
         consensus_seq = Seq(consensus_seq_str, alpha) 
      seq_record = SeqRecord(consensus_seq, id=ace_contig.name, name=ace_contig.name) 
      quals = [] 
      i = 0 
      for base in consensus_seq: 
         if (base == '-'): 
            quals.append(0) 
         else: 
            quals.append(ace_contig.quality[i]) 
            i += 1 
      assert (i == len(ace_contig.quality)) 
      seq_record.letter_annotations['phred_quality'] = quals 
      (yield seq_record)

def tz_from_string(_option, _opt_str, value, parser): 
    if (value is not None): 
      if (value[0] in ['+', '-']): 
         valarray = [value[i:(i + 2)] for i in range(1, len(value), 2)] 
         multipliers = [3600, 60] 
         offset = 0 
         for i in range(min(len(valarray), len(multipliers))): 
            offset += (int(valarray[i]) * multipliers[i]) 
         if (value[0] == '-'): 
            offset = (- offset) 
         timezone = OffsetTzInfo(offset=offset) 
      elif tz_pytz: 
         try: 
            timezone = pytz.timezone(value) 
         except pytz.UnknownTimeZoneError: 
            debug.error('Unknown   display   timezone   specified') 
      else: 
         if (not hasattr(time, 'tzset')): 
            debug.error("This   operating   system   doesn't   support   tzset,   please   either   specify   an   offset   (eg.   +1000)   or   install   pytz") 
         timezone = value 
      parser.values.tz = timezone

def nova_docstring_multiline_start(physical_line, previous_logical, tokens): 
    if is_docstring(physical_line, previous_logical): 
      pos = max([physical_line.find(i) for i in START_DOCSTRING_TRIPLE]) 
      if ((len(tokens) == 0) and (pos != (-1)) and (len(physical_line) == (pos + 4))): 
         if (physical_line.strip() in START_DOCSTRING_TRIPLE): 
            return (pos, 'N404:   multi   line   docstring   should   start   with   a   summary')

def test_grouped_item_access(T1): 
    for masked in (False, True): 
      t1 = Table(T1, masked=masked) 
      tg = t1.group_by('a') 
      tgs = tg[('a', 'c', 'd')] 
      assert np.all((tgs.groups.keys == tg.groups.keys)) 
      assert np.all((tgs.groups.indices == tg.groups.indices)) 
      tgsa = tgs.groups.aggregate(np.sum) 
      assert (tgsa.pformat() == ['   a         c            d   ', '---   ----   ---', '      0      0.0         4', '      1      6.0      18', '      2   22.0         6']) 
      tgs = tg[('c', 'd')] 
      assert np.all((tgs.groups.keys == tg.groups.keys)) 
      assert np.all((tgs.groups.indices == tg.groups.indices)) 
      tgsa = tgs.groups.aggregate(np.sum) 
      assert (tgsa.pformat() == ['   c            d   ', '----   ---', '   0.0         4', '   6.0      18', '22.0         6'])

def _find_bad_optimizations0(order, reasons, r_vals): 
    for (i, node) in enumerate(order): 
      for new_r in node.outputs: 
         for (reason, r, old_graph_str, new_graph_str) in reasons[new_r]: 
            new_r_val = r_vals[new_r] 
            r_val = r_vals[r] 
            assert (r.type == new_r.type) 
            if hasattr(new_r.tag, 'values_eq_approx'): 
               check = new_r.tag.values_eq_approx(r_val, new_r_val) 
            elif hasattr(new_r, 'values_eq_approx'): 
               check = new_r.values_eq_approx(r_val, new_r_val) 
            else: 
               check = r.type.values_eq_approx(r_val, new_r_val) 
            if (not check): 
               raise BadOptimization(old_r=r, new_r=new_r, old_r_val=r_val, new_r_val=new_r_val, reason=reason, old_graph=old_graph_str, new_graph=new_graph_str)

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

def fix_unix_encoding(folder): 
    if ((not sabnzbd.WIN32) and (not sabnzbd.DARWIN) and gUTF): 
      for (root, dirs, files) in os.walk(folder.encode('utf-8')): 
         for name in files: 
            new_name = special_fixer(name).encode('utf-8') 
            if (name != new_name): 
               try: 
                  shutil.move(os.path.join(root, name), os.path.join(root, new_name)) 
               except: 
                  logging.info('Cannot   correct   name   of   %s', os.path.join(root, name))

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

