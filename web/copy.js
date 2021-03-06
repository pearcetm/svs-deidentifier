function Copy(page,imports){	
	page=$(page);
	//"import" functions/objects from the main app script that calls this module
	var filesep = '/';
	var filedlg = imports.filedlg;
	var targetdlg = imports.targetdlg;
	var faileddlg = imports.faileddlg;	
	var byteval = imports.byteval;
	var overlay = imports.overlay;
	var uniqueID = imports.uniqueID;
	var get_files = imports.get_files;
	var get_dir = imports.get_dir;
	var set_destination = imports.set_destination;
	
	$('body').on('click','.progress-monitor.failed',function(e){
		var el = $(this);
		faileddlg.find('.error-message').text(el.data('failure-message'));
		faileddlg.data('target',el);
		overlay.trigger('activate',[faileddlg]);
	});
	// $('body').on('update-fileset-header','.fileset',function(e){
	// 	update_fileset_header($(this));
	// })
	function pick_destination(){
		overlay.trigger('activate',[filedlg])
		get_dir(set_destination)
	}
	eel.expose(update_progress)
	function update_progress(data){
		//console.log(data)
		$.each(data.destinations, function(i,e){
			// console.log('Destination set:',e);
			var el=page.find('#'+e.id)
			var d = el.find('.progress-dest');
			d.removeClass('requested-dest').addClass('actual-dest');
			d.find('.filepath').text(e.dest)
			if(e.renamed){
				d.addClass('renamed').attr('title','Warning: File already existed, copy was renamed');
			}
		});
		$.each(data.progress, function(i,e){
			// console.log('Progress set:',e);
			var el=page.find('#'+e.id)
			var p = el.find('.progress-report');
			var units=p.find('.total .byteval').data('units');
			var total=p.find('.total').data('filesize');
			p.find('.current').empty().append(byteval(e.progress,units)).data('remaining',total-e.progress);;
			p.find('.percent').text((100*e.progress/total).toFixed(0));
			p.find('progress').attr('value',e.progress/total);
		});
		$.each(data.finalized, function(i,e){
			// console.log('Finalized set:',e);
			var el=page.find('#'+e.id)
			el.addClass('finalized');
			var progressbar=el.find('progress').attr('value',e.failed?0:1);//update the progress bar
			if(e.failed){
				el.addClass('failed').data('failure-message',e.failure_message);
			}
			// if(data.finalized.length-1 == i){
			// 	update_fileset_header(el.closest('.fileset'));
			// }
		});
	}

	function progress_monitor(e){
		var p = $('<div>',{class:'columns progress-monitor',id:e.attr('id')});
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
		current.append(byteval(0, total.find('.byteval').data('units'))).data('remaining',filesize);
		percent.text('0');

		p.data('original',e);
		return p;
	}
	

	function do_deidentification(fileset){
		
		//make a list of the files to work on
		var files=fileset.find('.file');
		//re-order the DOM before starting the process
		files.prependTo(fileset.find('.filelist'));

		console.log('Deidentifying files:',files.length)
		var fileinfo=files.map(function(i,e){
			e=$(e);
			var basedest=fileset.find('.target').data('destination').directory + filesep;
			var id=uniqueID();
			e.attr('id',id);
			var source=e.find('.filepath').data('file');
			var dest=basedest+e.find('.filedest .dest').val();
			e.data('requested-destination',dest);//save now for accessing in progress element
			progress_monitor(e).insertAfter(e);
			e.detach();//do this instead of remove or replaceWith to preserve bound data
			return {source:source,dest:dest,id:id}
		}).toArray();
		// console.log('file info',fileinfo)
		// change status of the display to reflect ongoing copy/delabel operation
		fileset.addClass('inprogress').removeClass('finished');
		
		//setup timer for calculating estimated time remaining
		var size_remaining = fileset.find('.progress-monitor:not(.finalized) .total').toArray().reduce(function(t,e){
			return t+$(e).data('filesize');
		},0);
		var now = new Date().getTime();
		var starttime=now;
		if(!!fileset.data('started')) starttime=fileset.data('starttime');//save old value if new files were added while copying
		fileset.data({timer:[],starttime:starttime});//clear the old timer and set the start time 
		update_fileset_header(fileset);

		//start timing the operation
		var intervalfunc = ( (fs)=> ()=>update_fileset_header(fs))(fileset);//capture fileset into lambda for setInterval callback
		var interval=window.setInterval(intervalfunc, 1000);
		fileset.data('interval',interval);

		//disable changing the target directory now that we've started
		fileset.find('.change-target').prop('disabled',true);

		//do the job
		eel.do_copy_and_strip(fileinfo)
	}

	function add_new_fileset(fileinfo){
		//console.log('File info:', fileinfo)
		var fileset = add_files_to_fileset(fileinfo);
		if(fileinfo.aperio.length>0){
			targetdlg.data({fileset:fileset,fileinfo:fileinfo.aperio});
			overlay.trigger('activate',[targetdlg]);
		}
		// else{
		// 	overlay.trigger('inactivate');//this is handled in add_files_to_fileset
		// }
	}
	function update_fileset_header(fileset){
		var h = fileset.find('.fileset-header');
		var target = h.find('.target');
		var info=target.data('destination');
		
		var rs = fileset.find('.progress-monitor:not(.finalized) .current').toArray().reduce(function(t,e){
			return t+$(e).data('remaining');
		},0);
		var ts = fileset.find('.filesize').toArray().reduce(function(t,e){
			return t+$(e).data('filesize');
		}, rs);
		
		//calculate remaining time
		var now = new Date().getTime();
		var timer=fileset.data('timer');
		var starttime=fileset.data('starttime');
		if(!timer) timer=[];
		timer.push({t:now,s:rs});

		//the first few seconds will show too fast of progress because of disk cache writing
		//the estimate of total time will be much better if we ignore this.
		if( (now-starttime) < 7500) timer=timer.slice(-1);//only keep the most recent value

		if( (now-starttime) < 10500) timer=timer.slice(-2);//only keep the most recent 2 values for the next seconds

		var timer=timer.slice(-20);//only keep up to 20 data points
		fileset.data('timer',timer);

		var remaining=fileset.find('.time-remaining');
		var progress=timer[0].s - timer.slice(-1)[0].s; //size at start minus size now
		var interval=timer.slice(-1)[0].t - timer[0].t; //time now minus time at start
		

		if(interval < 3500){
			//make sure we have enough data points for a reasonable calculation
			remaining.text('Calculating...');
		}
		else{
			var rate=progress/interval;
			var est = rs / rate; //estimated time remaining in milliseconds
			var totalsec = est/1000;//convert to seconds
			var hours=Math.floor(totalsec / 3600);
			var minutes=Math.floor( (totalsec - (hours*3600)) / 60);
			var seconds=Math.floor(totalsec - (hours*3600) - (minutes*60));
			var formatted = ('00'+seconds).slice(-2)+'s';
			if( (minutes+hours) > 0) formatted = ''+minutes+'m '+formatted;
			if( hours > 0) formatted = ''+hours+'h '+formatted;
			remaining.text(formatted);
			//console.log('Rate',rate/1000000,'Remaining',rs/1000000)
		}

		if(rs==0 && fileset.hasClass('inprogress')){
			fileset.removeClass('inprogress').addClass('finished');
			window.clearInterval(fileset.data('interval'));
			var totalsec = (now - fileset.data('starttime')) / 1000;//convert to seconds
			var hours=Math.floor(totalsec / 3600);
			var minutes=Math.floor( (totalsec - (hours*3600)) / 60);
			var seconds=Math.floor(totalsec - (hours*3600) - (minutes*60));
			var formatted = ('00'+seconds).slice(-2)+'s';
			if( (minutes+hours) > 0) formatted = ''+minutes+'m '+formatted;
			if( hours > 0) formatted = ''+hours+'h '+formatted;
			remaining.text(formatted);
		}

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