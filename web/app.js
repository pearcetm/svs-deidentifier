$(function(){

	var filesep = '/';
	var overlay=$('.overlay');
	var targetdlg=$('.overlay-dialog.select-target');
	var filedlg=$('.overlay-dialog.choose-file');
	var faileddlg=$('.overlay-dialog.failed');
	var optionsdlg=$('.overlay-dialog.options');
	var helpdlg=$('.overlay-dialog.help');

	$('#pick-files').on('click',function(){
		overlay.trigger('activate',[filedlg])
		get_files(add_new_fileset)
	})
	$('.pick-destination').on('click',function(){
		overlay.trigger('activate',[filedlg])
		get_dir(set_destination)
	})
	$('#do-deidentification').on('click',function(){
		do_deidentification()
	})
	$('.dest-selection .remove-button').on('click',function(){
		$(this).parent().removeAttr('selected').find('.destination').empty();
		targetdlg.find('.ok').attr('disabled',true);
	});
	$('body').on('click','.progress-monitor.failed',function(e){
		var el = $(this);
		faileddlg.find('.error-message').text(el.data('failure-message'));
		faileddlg.data('target',el);
		overlay.trigger('activate',[faileddlg]);
	});
	

	// Defer final setup until end of current event loop
	setTimeout(function(){
		overlay.trigger('activate',[optionsdlg])
	},0)
	
	
	function get_files(callback){
		var dlgtype=localStorage.getItem('fd-type')
		eel.get_files(dlgtype,get_source_startpath())(callback)
	}
	function get_dir(callback){
		var dlgtype=localStorage.getItem('fd-type')
		eel.get_dir(dlgtype,get_dest_startpath())(callback)
	}

	eel.expose(update_progress)
	function update_progress(data){
		console.log(data)
		$.each(data.destinations, function(i,e){
			// console.log('Destination set:',e);
			var el=$('#'+e.id)
			var d = el.find('.progress-dest');
			d.removeClass('requested-dest').addClass('actual-dest');
			d.find('.filepath').text(e.dest)
			if(e.renamed){
				d.addClass('renamed').attr('title','Warning: File already existed, copy was renamed');
			}
		});
		$.each(data.progress, function(i,e){
			// console.log('Progress set:',e);
			var el=$('#'+e.id)
			var p = el.find('.progress-report');
			var units=p.find('.total .byteval').data('units');
			var total=p.find('.total').data('filesize');
			p.find('.current').empty().append(byteval(e.progress,units)).data('remaining',total-e.progress);;
			p.find('.percent').text((100*e.progress/total).toFixed(0));
			p.find('progress').attr('value',e.progress/total);
		});
		$.each(data.finalized, function(i,e){
			// console.log('Finalized set:',e);
			var el=$('#'+e.id)
			el.addClass('finalized');
			var progressbar=el.find('progress').attr('value',e.failed?0:1);//update the progress bar
			if(e.failed){
				el.addClass('failed').data('failure-message',e.failure_message);
			}
			if(data.finalized.length-1 == i){
				update_fileset_header(el.closest('.fileset'));
			}
		});
	}

	function progress_monitor(e){
		var p = $('<div>',{class:'columns progress-monitor',id:e.attr('id')});
		// $('<progress>').appendTo(p);
		var src=$('<div>',{class:'progress-src left-ellipsis-outer'}).appendTo(p);
		$('<span>',{class:'filepath left-ellipsis-inner'}).appendTo(src).text(e.find('.sourcefile').data('file'))
		var rep = $('<div>',{class:'progress-report'}).appendTo(p)
		var current = $('<span>',{class:'current update'}).appendTo(rep);
		var total = $('<span>',{class:'total update'}).appendTo(rep);
		var percent = $('<span>',{class:'percent update'}).appendTo(rep);
		var dest=$('<div>',{class:'progress-dest requested-dest left-ellipsis-outer'}).appendTo(p);
		$('<span>',{class:'filepath left-ellipsis-inner'}).appendTo(dest).text(e.data('requested-destination'));

		$('<progress>',{max:1,value:0}).prependTo(rep);

		var filesize=e.find('.filesize').data('filesize');
		total.append(byteval(filesize)).data('filesize',filesize);
		current.append(byteval(0, total.find('.byteval').data('units')));
		percent.text('0');

		p.data('original',e);
		return p;
	}
	

	function do_deidentification(){
		//re-order the DOM before starting the process
		$('.fileset:not(.invalid)').each(function(i,e){
			$(e).find('.file').prependTo($(e).find('.filelist'));
		})
		var files=$('.fileset:not(.invalid) .file');
		console.log('Deidentifying files:',files.length)
		var fileinfo=files.filter(function(i,e){
			//use the presence of a (non-empty, non-undefined) directory to filter invalid files
			return $(e).closest('.fileset').find('.target').data('destination') !== undefined;
		}).map(function(i,e){
			e=$(e);
			var basedest=e.closest('.fileset').find('.target').data('destination').directory + filesep;
			var id=uniqueID();
			e.attr('id',id);
			var source=e.find('.filepath').data('file');
			var dest=basedest+e.find('.filedest .dest').val();
			e.data('requested-destination',dest);//save now for accessing in progress element
			//e.replaceWith(progress_monitor(e));//replace with element to monitor progress of copy operation
			progress_monitor(e).insertAfter(e);
			e.detach();//do this instead of remove or replaceWith to preserve bound data
			return {source:source,dest:dest,id:id}
		}).toArray();
		console.log('file info',fileinfo)
		eel.do_copy_and_strip(fileinfo)
	}

	function add_new_fileset(fileinfo){
		console.log('File info:', fileinfo)
		if(fileinfo.invalid.length>0){
			invalid_files(fileinfo.invalid);
		}
		var fileset = setup_fileset(fileinfo.aperio)
		//targetdlg.trigger('set-fileinfo',[fileinfo.aperio]);
		targetdlg.data({fileset:fileset,fileinfo:fileinfo.aperio});
		$('.overlay').trigger('activate',[targetdlg]);

	}
	function update_fileset_header(fileset){
		var h = fileset.find('.fileset-header');
		var target = h.find('.target');
		var info=target.data('destination');
		var ts = fileset.find('.filesize').toArray().reduce(function(t,e){
			return t+$(e).data('filesize');
		}, 0)
		ts = fileset.find('.progress-monitor:not(.finalized) .current').toArray().reduce(function(t,e){
			return t+$(e).data('remaining');
		},ts);
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
	}
	function setup_fileset(list){
		
		var fileset = $('<div>',{class:'fileset'}).appendTo('#filelist');
		var h = $('<div>',{class:'columns fileset-header'}).appendTo(fileset);

		var h1=$('<div>',{class:'oneline'}).appendTo(h).append($('<h4>').html('Files to copy: <span class="num-files"></span>'));
		var h2=$('<div>',{class:'columns target-field'}).appendTo(h);
		$('<h4>',{class:'left-ellipsis-outer'}).appendTo(h2)
				.append($('<span>',{class:"left-ellipsis-inner target filepath"}));


		var h3=$('<div>',{class:'oneline'}).appendTo(h2).append($('<h4>').html(
			'Total size:<span class="totalsize"></span> (<span class="freespace"></span> available)'));

		$('<button>',{class:'add-files'}).text('+').on('click',function(){
			overlay.trigger('activate',[filedlg])
			get_files((function(f){return function(files){add_files_to_fileset(files,f)} })(fileset))
		}).insertAfter(h1.find('.num-files'));

		$('<button>',{class:'change-target'}).text('Choose').on('click',function(){
			overlay.trigger('activate',[filedlg])
			get_dir((function(f){return function(d){
				if(d.directory !==''){
					set_fileset_destination(f,d);
				}
				overlay.trigger('inactivate');
			} })(fileset))
		}).insertBefore(h3);

		$('<button>',{class:'clear-fileset remove-button'}).text('X').on('click',function(){
			fileset.remove();
		})
		.appendTo(fileset);
		
		var filelist=$('<div>',{class:'filelist'}).appendTo(fileset);
		$.each(list, function(i,v){
			make_file_row(v).appendTo(filelist);
		})
		update_fileset_header(fileset);
		return fileset;
	}
	function add_files_to_fileset(fileinfo,fileset){
		if(fileinfo.invalid.length>0){
			invalid_files(fileinfo.invalid);
		}
		$('.overlay').trigger('inactivate');
		$.each(fileinfo.aperio, function(i,v){
			make_file_row(v).appendTo(fileset.find('.filelist'));
		})
		update_fileset_header(fileset);
	}
	function make_file_row(v){
		var row=$('<div>',{class:'columns file'});
		var l = $('<div>',{class:'columns source'}).appendTo(row);
		var r = $('<div>',{class:'columns'}).appendTo($('<div>').appendTo(row));
		var leo=$('<div>',{class:'left-ellipsis-outer'}).appendTo(l);
		$('<div>',{class:'filepath sourcefile left-ellipsis-inner'}).text(v.file).data('file',v.file).appendTo(leo);
		$('<div>',{class:'filesize'}).append(byteval(v.filesize)).data('filesize',v.filesize).appendTo(l)
		var fd=$('<div>',{class:'filedest'}).appendTo(r);
		$('<input>',{class:'dest', placeholder:'Enter a file name'}).val(v.destination).appendTo(fd)
		$('<button>',{class:'remove-button', title:'Remove item'})
			.text('X').appendTo(r).on('click',function(){
				var fileset=row.closest('.fileset')
				row.remove();
				update_fileset_header(fileset);
			})

		return row;
	}
	function invalid_files(list){
		var fileset = $('<div>',{class:'fileset invalid'}).appendTo('#filelist');
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
	function set_destination(dir_info){
		targetdlg.trigger('destination-selected',[dir_info]);
	}
	function make_list(){
		$('#filelist .file').map(function(i,e){
			console.log(i,e);

			var src=$(e).find('.filepath').data('file');
			var dest=$('#destination').data('path')+$(e).find('.dest').val()
			return {
				source:src,
				destination:dest
			}
		})
	}
	function byteval(bytes,units='auto',decimal_places=1){
		var d = $('<span>',{class:'byteval'});
		var a = $('<span>',{class:'bytes base-1000'}).appendTo(d);
		var b = $('<span>',{class:'bytes base-1024'}).appendTo(d);

		var calc=['B','KB','MB','GB','TB'].map(function(u,exp){
			return{
				m1000:bytes/(1000**exp),
				m1024:bytes/(1024**exp),
				units:u,
			}
		}).reverse().find(function(v){
			if(units=='auto'){
				return v.m1024 >= 1;
			}
			else{
				return v.units==units;
			}
		});
		if(calc){
			a.text(calc.m1000.toFixed(decimal_places));
			b.text(calc.m1024.toFixed(decimal_places));
			d.attr('data-units',calc.units);
		}
		else if(bytes==0){
			a.text(0);
			b.text(0);
			d.attr('data-units',units=='auto'?'B':units);
		}
		return d;
		// convert bytes to megabytes rounded to defined decimal place (default = 1)
		//return (n/(1024*1024)).toFixed(decimal_places);
	}

	overlay.on('activate',function(ev,dialog){
		if(ev.target !== this) return;
		console.log('activating overlay')
		// console.log('Activating overlay',ev,dialog)
		overlay.addClass('active');
		overlay.find('.overlay-content .overlay-dialog').appendTo(overlay.find('.overlay-dialogs'))
		overlay.find('.overlay-content').append(dialog);
		dialog.trigger('activate');
	});
	overlay.on('inactivate',function(){
		overlay.removeClass('active');
	})
	// $('.overlay-background').on('click',function(ev){
	// 	if(ev.target == this){
	// 		overlay.trigger('inactivate');
	// 	}
	// });
	targetdlg.on('activate',function(){
		var list = targetdlg.data('fileinfo');
		var num = targetdlg.find('.num-files').text(list.length);
		if(list.length==1){
			num.parent().removeClass('pluralize');
		}
		else{
			num.parent().addClass('pluralize');
		}
		targetdlg.find('.ok').attr('disabled',true);
		targetdlg.find('.dest-selection').removeAttr('selected').find('.destination').empty();
		if(list.length==0) overlay.trigger('inactivate'); //close the dialog if user cancels the operation
	})
	targetdlg.on('destination-selected',function(ev,dir_info){
		overlay.trigger('activate',[targetdlg]);
		if(dir_info.directory !== '' && dir_info.writable){
			//only enable the "OK" button if a WRITABLE directory is selected
			targetdlg.find('.dest-selection').attr('selected',true);
			targetdlg.find('.ok').attr('disabled',false);
			targetdlg.find('.destination').text(dir_info.directory);
			targetdlg.data({destination:dir_info.directory,total:dir_info.total,free:dir_info.free});
		}
		else if(dir_info.directory !== ''){
			targetdlg.find('.dest-selection').attr('selected',true);
			targetdlg.find('.ok').attr('disabled',true);
			var err=$('<span>',{class:'error-message'}).text('This directory is read-only');
			var errdiv=$('<div>').append(err);
			targetdlg.find('.destination').text(dir_info.directory).append(errdiv);
		}
		else{
			targetdlg.find('.dest-selection').removeAttr('selected');
			targetdlg.find('.ok').attr('disabled',true);
		}
	})
	targetdlg.find('.ok').on('click', function(){
		overlay.trigger('inactivate');
		var fileset = targetdlg.data('fileset');
		var d = targetdlg.data();
		var dest = {
			directory:d.destination,
			total:d.total,
			free:d.free
		}
		set_fileset_destination(fileset,dest);
	})
	targetdlg.find('.cancel').on('click', function(){
		overlay.trigger('inactivate');
	})

	faileddlg.find('.ok').on('click',function(){
		var el = faileddlg.data('target');
		var file_item = el.data('original');
		//el.replaceWith(file_item);
		file_item.appendTo(el.closest('.filelist'));
		el.remove();
		overlay.trigger('inactivate');
		update_fileset_header(file_item.closest('.fileset'));
	})
	faileddlg.find('.cancel').on('click', function(){
		overlay.trigger('inactivate');
	})
	optionsdlg.find('.ok').on('click',function(){
		overlay.trigger('inactivate');
	})
	var uniqueID = (function(){
	    var i=0;
	    return function() {
	        return 'uniqueID_'+(i++);
	    }
	})();

	async function test_file_dialog(type){
		// $('body').attr('disabled',true)
		// console.log('testing file dialog type='+type)
		$('body').addClass('no-pointer-events');
		var output = await eel.test_file_dialog(type)()
		console.log('finished test', output)
		$('body').removeClass('no-pointer-events')
		
		// $('body').attr('disabled',false)
	}
	async function get_config_path(path, callback){
		$('body').addClass('no-pointer-events');
		var output = await eel.get_config_path(localStorage.getItem('fd-type'),path)()
		console.log('finished getting config path', output)
		$('body').removeClass('no-pointer-events')
		callback(output)
	}
	function setup_options_dialog(){
		console.log('setup_option_dialog called')
		$('.options .radio').each(function(i,e){
			$(e).prepend(
			$('<input>',{type:'radio',required:true}).on('change',function(){
				// console.log('change')
				var d = $(this).data()
				localStorage.setItem(d.store,d.val)
			}).attr('name',$(e).data('store')).data($(e).data()) ); 
		});
		$('.options [data-store="fd-type"]')
			.append($('<span>',{class:'test-fd'}).text('[test]').on('click',function(){
				// console.log('click')
				var type=$(this).closest('[data-store="fd-type"]').data('val');
				test_file_dialog(type)
			}));
		$('.options [data-store^="fd-stay-"]')
			.append($('<span>',{class:'test-fd'}).text('[choose]').on('click',function(){
				var p=$(this).closest('[data-store]')
				var store=p.data('store');
				var dir = get_config_path(localStorage.getItem(store),function(dir){
					localStorage.setItem(store,dir.absolutePath)
					console.log('setting',store,dir.absolutePath)
					p.find('.selected-path').text(dir.absolutePath);
				});
				
			}));
		$('.options .path').each(function(){
			var p = $(this);
			var store=p.data('store');
			var dir = localStorage.getItem(store);
			p.prepend($('<span>',{class:'selected-path'}).text(dir?dir:''));
		});
		var groups=$('.options input:radio').toArray().reduce(function(a,e){
			if(a.indexOf(e.name)==-1) a.push(e.name);
			return a;
		},[]);
		var unset_status = groups.map(function(e){
			var unset = $('.options input:radio[name="'+e+'"]')
				.filter(function(i,e){
					var d = $(e).data();
					if(localStorage.getItem(d.store)==d.val){
						console.log('setting '+d.store+' from localStorage to '+d.val)
						$(e).prop('checked',true);
						return true;
					}
					return false; 
				}).length==0
			if(unset){
				var r = $('.options input:radio[name="'+e+'"]');
				console.log('setting first item to checked')
				r.first().click()
			}
			return unset;
		});
		var any_unset=unset_status.filter(function(e){return e==true;}).length>0
		if(any_unset){
			$('.options .options-message').addClass('unset')
		}
		else{
			$('.options .options-message').removeClass('unset')
		}
	}
	
	function refreshoptions(){
		$('.overlay-dialog.options .markdown').load('/options',null,setup_options_dialog);
	}
	$.ajaxSetup ({
	    // Disable caching of AJAX responses
	    cache: false
	});
	refreshoptions()
	//Local storage for persistence of user choices

	eel.expose(set_follow_dest)
	function set_follow_dest(dir){
		localStorage.setItem('fd-follow-dest',dir);
	}
	eel.expose(set_follow_source)
	function set_follow_source(dir){
		localStorage.setItem('fd-follow-source',dir)
	}
	function get_dest_startpath(){
		var beh = localStorage.getItem('fd-behavior')
		var path = localStorage.getItem('fd-'+beh+'-dest')
		return path?path:'';
	}
	function get_source_startpath(){
		var beh = localStorage.getItem('fd-behavior')
		var path = localStorage.getItem('fd-'+beh+'-source')
		return path?path:'';
	}
})

