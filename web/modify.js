function Modify(page,imports){	
	page=$(page);
	//"import" functions/objects from the main app script that calls this module
	var filesep = '/';
	var filedlg = imports.filedlg;
	var targetdlg = imports.targetdlg;	
	var byteval = imports.byteval;
	var overlay = imports.overlay;
	var uniqueID = imports.uniqueID;
	var get_files = imports.get_files;
	var get_dir = imports.get_dir;
	var set_destination = imports.set_destination;
	

	// function pick_destination(){
	// 	overlay.trigger('activate',[filedlg])
	// 	get_dir(set_destination)
	// }
	eel.expose(update_progress)
	function update_modify_process(data){
		//console.log(data)
		
		$.each(data.finalized, function(i,e){
			// console.log('Finalized set:',e);
			var el=page.find('#'+e.id)
			el.addClass('modified delabeled');
			if(e.failed){
				el.addClass('failed').data('failure-message',e.failure_message);
			}
		});
		update_fileset_header(el.closest('.fileset'));
	}

	// function progress_monitor(e){
	// 	var p = $('<div>',{class:'columns progress-monitor',id:e.attr('id')});
	// 	var src=$('<div>',{class:'progress-src left-ellipsis-outer'}).appendTo(p);
	// 	$('<span>',{class:'filepath left-ellipsis-inner'}).appendTo(src).text(e.find('.sourcefile').data('file'))
	// 	var rep = $('<div>',{class:'progress-report'}).appendTo(p)
	// 	var current = $('<span>',{class:'current update'}).appendTo(rep);
	// 	var total = $('<span>',{class:'total update'}).appendTo(rep);
	// 	var percent = $('<span>',{class:'percent update'}).appendTo(rep);
	// 	var dest=$('<div>',{class:'progress-dest requested-dest left-ellipsis-outer'}).appendTo(p);
	// 	$('<span>',{class:'filepath left-ellipsis-inner'}).appendTo(dest).text(e.data('requested-destination'));

	// 	$('<progress>',{max:1,value:0}).prependTo(rep);

	// 	var filesize=e.find('.filesize').data('filesize');
	// 	total.append(byteval(filesize)).data('filesize',filesize);
	// 	current.append(byteval(0, total.find('.byteval').data('units')));
	// 	percent.text('0');

	// 	p.data('original',e);
	// 	return p;
	// }
	

	function do_deidentification(fileset){
		
		//make a list of the files to work on
		var files=fileset.find('.file:not(.modified)');
		//re-order the DOM before starting the process
		files.prependTo(fileset.find('.filelist'));

		console.log('Deidentifying files:',files.length)
		var fileinfo=files.map(function(i,e){
			e=$(e);
			var basedest=fileset.find('.target').data('destination').directory + filesep;
			var id=uniqueID();
			e.attr('id',id);
			var source=e.find('.filepath').data('file');
			return {source:source,id:id}
		}).toArray();
		// console.log('file info',fileinfo)
		eel.do_copy_and_strip(fileinfo);
	}

	function add_new_fileset(fileinfo){
		//console.log('File info:', fileinfo)
		var fileset = add_files_to_fileset(fileinfo);		
	}
	function update_fileset_header(fileset){
		var h = fileset.find('.fileset-header');
		
		h.find('.totalsize').empty().append(byteval(ts));
		h.find('.num-files').text(fileset.find('.file').length);
		if(info){
			//var freespace = info.free;
			//h.find('.freespace').empty().append(byteval(freespace));
			eel.check_free_space(info.directory)(function(free){set_freespace(fileset,free)});
			target.text(info.directory);
		}
	}
	function set_freespace(fileset,freespace){
		fileset.find('.freespace').empty().append(byteval(freespace));
	}
	function set_fileset_destination(fileset,destination){
		fileset.find('.fileset-header .target').data('destination',destination);
		update_fileset_header(fileset);
		if(!!destination){
			//enable the button to initiate de-identification if a destination has been set
			fileset.find('.deidentify-fileset').attr('disabled',false);
		}
	}
	function setup_fileset(fileinfo){
		var fileset = page.find('.templates .fileset').clone().appendTo(page.find('.filesetlist'));
		fileset.find('.add-files').on('click',function(){
			overlay.trigger('activate',[filedlg])
			get_files((function(f){return function(files){add_files_to_fileset(files,f)} })(fileset))
		});
		fileset.find('.change-target').on('click',function(){
			overlay.trigger('activate',[filedlg])
			get_dir((function(f){return function(d){
				if(d.directory !==''){
					set_fileset_destination(f,d);
				}
				overlay.trigger('inactivate');
			} })(fileset))
		});
		fileset.find('.clear-fileset').on('click',function(){
			fileset.remove();
		});
		
		return fileset;
	}
	function add_files_to_fileset(fileinfo,fileset){
		overlay.trigger('inactivate');
		//do invalid files first so they are on top and visible
		if(fileinfo.invalid.length>0){
			invalid_files(fileinfo.invalid);
		}
		if(fileinfo.aperio.length>0){
			if(typeof fileset == 'undefined'){
				fileset = setup_fileset();
			}
			$.each(fileinfo.aperio, function(i,v){
				make_file_row(v).appendTo(fileset.find('.filelist'));
			});
			update_fileset_header(fileset);
		}
		return fileset;
	}
	function make_file_row(v){
		var row = page.find('.templates .file').clone();
		row.find('.sourcefile').text(v.file).data('file',v.file);
		row.find('.filesize').append(byteval(v.filesize)).data('filesize',v.filesize);
		row.find('input.dest').val(v.destination);
		row.find('.remove-button').on('click',function(){
			var fileset=row.closest('.fileset')
			row.remove();
			update_fileset_header(fileset);
		})

		return row;
	}
	function invalid_files(list){
		var fileset = $('<div>',{class:'fileset invalid'}).appendTo(page.find('.filesetlist'));
		var h = $('<div>',{class:'columns fileset-header invalid'}).appendTo(fileset);
		var l = $('<div>').appendTo(h);
		var r = $('<div>').appendTo(h);
		var clear=$('<button>',{class:'clear-invalid clear-fileset'}).text('Remove').appendTo(r);
		clear.on('click',function(){fileset.remove()});

		$('<h4>').html('Invalid files detected').appendTo(l);
		
		$.each(list, function(i,v){
			var row=$('<div>',{class:'columns file filepath invalid'}).appendTo(fileset);
			row.text(v.file)
		})
	}

	page.find('.pick-files').on('click',function(){
		overlay.trigger('activate',[filedlg]);
		targetdlg.data('pick-destination-function',pick_destination);
		targetdlg.data('set-destination-function',set_fileset_destination);
		get_files(add_new_fileset);
	});
	page.find('.pick-destination').on('click',pick_destination)
	page.on('click','.deidentify-fileset',function(){
		var fileset = $(this).closest('.fileset');
		do_deidentification(fileset);
	});

	return this;
}