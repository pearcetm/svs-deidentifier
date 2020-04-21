function Modify(page,imports){	
	page=$(page);
	//"import" functions/objects from the main app script that calls this module
	var filesep = '/';
	var filedlg = imports.filedlg;
	var targetdlg = imports.targetdlg;
	var confirmdlg = imports.confirmdlg;	
	var byteval = imports.byteval;
	var overlay = imports.overlay;
	var uniqueID = imports.uniqueID;
	var get_inplace_files = imports.get_inplace_files;
	var get_inplace_dir = imports.get_inplace_dir;
	var set_destination = imports.set_destination;
	
	confirmdlg.find('.cancel').on('click',function(){overlay.trigger('inactivate')});

	
	function ask_confirmation(fileset){
		confirmdlg.find('.ok').off('click').on('click',function(){
			overlay.trigger('inactivate');
			do_deidentification(fileset);
		});
		overlay.trigger('activate',[confirmdlg]);
	}

	function do_deidentification(fileset){
		
		//make a list of the files to work on
		var files=fileset.find('.file:not(.modified)');
		//re-order the DOM before starting the process
		files.prependTo(fileset.find('.filelist'));

		console.log('Deidentifying files:',files.length)
		fileset.find('.modify-results').show();
		fileset.find('.deidentify-fileset').attr('disabled',true);//prevent double clicks
		var fileinfo=files.each(function(i,e){
			e=$(e);
			var source=e.data('file');
			console.log('stripping',source);
			eel.do_strip_in_place(source)( (function(fobj,src,fs){
				return function(status){
					console.log('done with',src,source)
					fobj.addClass('modified');
					if(status=='ok')fobj.removeClass('has-label').removeClass('has-macro');
					else fobj.addClass('failed').data('failure-message',status);
					update_fileset_header(fs);
				}
			})(e,source,fileset));
		});
		// console.log('file info',fileinfo)
		
	}

	function add_new_fileset(fileinfo){
		//console.log('File info:', fileinfo)
		var fileset = add_files_to_fileset(fileinfo);		
	}
	function update_fileset_header(fileset){
		var h = fileset.find('.fileset-header');
		h.find('.num-files').text(fileset.find('.file:not(.modified)').length);
		h.find('.num-both').text(fileset.find('.file:not(.modified).has-label.has-macro').length);
		h.find('.num-labels').text(fileset.find('.file:not(.modified).has-label:not(.has-macro)').length);
		h.find('.num-macros').text(fileset.find('.file:not(.modified).has-macro:not(.has-label)').length);
		h.find('.num-neither').text(fileset.find('.file:not(.modified):not(.has-label):not(.has-macro)').length);
		h.find('.num-succeeded').text(fileset.find('.file.modified:not(.failed)').length);
		h.find('.num-failed').text(fileset.find('.file.failed').length);
		
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
			get_inplace_files((function(f){return function(files){add_files_to_fileset(files,f)} })(fileset))
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
		if(fileinfo.readonly.length>0){
			readonly_files(fileinfo.invalid);
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
		var row = page.find('.templates .file').clone().data('file',v.file);
		if(v.has_label){
			row.addClass('has-label');
		}
		if(v.has_macro){
			row.addClass('has-macro');
		}
		row.find('.sourcefile').text(v.file);
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
	function readonly_files(list){
		var fileset = $('<div>',{class:'fileset readonly invalid'}).appendTo(page.find('.filesetlist'));
		var h = $('<div>',{class:'columns fileset-header readonly invalid'}).appendTo(fileset);
		var l = $('<div>').appendTo(h);
		var r = $('<div>').appendTo(h);
		var clear=$('<button>',{class:'clear-invalid clear-fileset'}).text('Remove').appendTo(r);
		clear.on('click',function(){fileset.remove()});

		$('<h4>').html('The following files are READ ONLY and cannot be modified').appendTo(l);
		
		$.each(list, function(i,v){
			var row=$('<div>',{class:'columns file filepath invalid'}).appendTo(fileset);
			row.text(v.file)
		})
	}

	page.find('.pick-files').on('click',function(){
		overlay.trigger('activate',[filedlg]);
		get_inplace_files(add_new_fileset);
	});
	page.find('.pick-folder').on('click',function(){
		overlay.trigger('activate',[filedlg]);
		get_inplace_dir(add_new_fileset);
	});
	
	page.on('click','.deidentify-fileset',function(){
		var fileset = $(this).closest('.fileset');
		ask_confirmation(fileset);
	});

	return this;
}