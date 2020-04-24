$(function(){

	var overlay=$('.overlay');
	var targetdlg=$('.overlay-dialog.select-target');
	var filedlg=$('.overlay-dialog.choose-file');
	var faileddlg=$('.overlay-dialog.failed');
	var confirmdlg=$('.overlay-dialog.confirm');
	

	$('.dest-selection .remove-button').on('click',function(){
		$(this).parent().removeAttr('selected').find('.destination').empty();
		targetdlg.find('.ok').attr('disabled',true);
	});
	
	

	function get_files(callback){
		var dlgtype=localStorage.getItem('fd-type')
		eel.get_files(dlgtype,get_source_startpath())(callback)
	}
	function get_dir(callback){
		var dlgtype=localStorage.getItem('fd-type')
		eel.get_dir(dlgtype,get_dest_startpath())(callback)
	}
	function get_inplace_files(callback){
		var dlgtype=localStorage.getItem('fd-type')
		eel.get_inplace_files(dlgtype,get_source_startpath())(callback)
	}
	function get_inplace_dir(callback){
		var dlgtype=localStorage.getItem('fd-type')
		eel.get_inplace_dir(dlgtype,get_dest_startpath())(callback)
	}

	
	function set_destination(dir_info){
		targetdlg.trigger('destination-selected',[dir_info]);
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
		
	}

	overlay.on('activate',function(ev,dialog){
		if(ev.target !== this) return;
		overlay.addClass('active');
		overlay.find('.overlay-content .overlay-dialog').appendTo(overlay.find('.overlay-dialogs'))
		overlay.find('.overlay-content').append(dialog);
		dialog.trigger('activate');
	});
	overlay.on('inactivate',function(){
		overlay.removeClass('active');
	})
	
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
		var func=targetdlg.data('set-destination-function');
		if(func){
			var fileset = targetdlg.data('fileset');
			var d = targetdlg.data();
			var dest = {
				directory:d.destination,
				total:d.total,
				free:d.free
			}
			func(fileset,dest);
		}
		else{
			console.log('set-destination-function not set on targetdlg');
		}
	})
	targetdlg.find('.cancel').on('click', function(){
		overlay.trigger('inactivate');
	})
	targetdlg.find('.pick-destination').on('click', function(){
		var func=targetdlg.data('pick-destination-function');
		if(func){
			func();
		}
		else{
			console.log('pick-destination-function not set on targetdlg');
		}
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
	function setup_settings_dialog(){
		//console.log('setup_option_dialog called')
		$('#settings .radio').each(function(i,e){
			$(e).prepend(
			$('<input>',{type:'radio',required:true}).on('change',function(){
				// console.log('change')
				var d = $(this).data()
				localStorage.setItem(d.store,d.val)
			}).attr('name',$(e).data('store')).data($(e).data()) ); 
		});
		$('#settings [data-store="fd-type"]')
			.append($('<span>',{class:'test-fd'}).text('[test]').on('click',function(){
				// console.log('click')
				var type=$(this).closest('[data-store="fd-type"]').data('val');
				test_file_dialog(type)
			}));
		$('#settings [data-store^="fd-stay-"]')
			.append($('<span>',{class:'test-fd'}).text('[choose]').on('click',function(){
				var p=$(this).closest('[data-store]')
				var store=p.data('store');
				var dir = get_config_path(localStorage.getItem(store),function(dir){
					localStorage.setItem(store,dir.absolutePath)
					console.log('setting',store,dir.absolutePath)
					p.find('.selected-path').text(dir.absolutePath);
				});
				
			}));
		$('#settings .path').each(function(){
			var p = $(this);
			var store=p.data('store');
			var dir = localStorage.getItem(store);
			p.prepend($('<span>',{class:'selected-path'}).text(dir?dir:''));
		});
		var groups=$('#settings input:radio').toArray().reduce(function(a,e){
			if(a.indexOf(e.name)==-1) a.push(e.name);
			return a;
		},[]);
		var unset_status = groups.map(function(e){
			var unset = $('#settings input:radio[name="'+e+'"]')
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
				var r = $('#settings input:radio[name="'+e+'"]');
				console.log('setting first item to checked')
				r.first().click()
			}
			return unset;
		});
		var any_unset=unset_status.filter(function(e){return e==true;}).length>0
		if(any_unset){
			$('#settings .settings-message').addClass('unset');
			// Defer switching to the settings tab until end of current event loop
			//setTimeout(opensettings,0)
		}
		else{
			$('#settings .settings-message').removeClass('unset')
		}
	}

	//toolbar mutation watchers - to allow bootstrap tabs with collapsible navbar
	$('.nav-tabs .nav-link').each(function(i,e){
		var observer = new MutationObserver(function(mutlist){
			//console.log('nav link changed',e,mutlist);
			var classchanged=mutlist.filter(function(e){return e.attributeName=='class'}).length>0;
			if(classchanged && $(e).hasClass('active') ){
				$('.nav-tabs .nav-link').not(e).removeClass('active');
			}
		});
		observer.observe(e,{attributes:true});
	})
	
		
	$.ajaxSetup ({
	    // Disable caching of AJAX responses
	    cache: false
	});
	$('#settings .markdown').load('/settings',null,setup_settings_dialog);
	$.ajax({
		type:'GET',
		url:'/readme', 
		success:function(d){
			$('#readme .markdown.help').html(d.help);
			$('#readme .markdown.readme').html(d.readme);
			$('#readme a[href^="#"]').attr('target','_self');
		}
	});

	$('#copy-tab-contents').appendTo('#copy-and-delabel');
	$('#modify-tab-contents').appendTo('#delabel-in-place');
	$('#home-tab-contents').appendTo('#home');

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




	//Call other javascript file objects for individual interface setup
	var exportvars = {
		byteval:byteval,
		overlay:$('.overlay'),
		uniqueID:uniqueID,
		filedlg:filedlg,
		faileddlg:faileddlg,
		targetdlg:targetdlg,
		confirmdlg:confirmdlg,
		get_files:get_files,
		get_dir:get_dir,
		get_inplace_files:get_inplace_files,
		get_inplace_dir:get_inplace_dir,
		set_destination,
	}
	Copy('#copy-and-delabel',exportvars);//setup copy interface
	Modify('#delabel-in-place',exportvars);//setup copy interface

})

