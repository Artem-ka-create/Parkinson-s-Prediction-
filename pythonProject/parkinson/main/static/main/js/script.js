

document.addEventListener("DOMContentLoaded",function(){
    'use strict';


    ///language

   let select = document.getElementsByClassName('select')[0];
   let select_head = document.getElementsByClassName('select__head')[0];
   let select_list = document.getElementsByClassName('select__list')[0];
   let file = document.getElementById(upload_button);

   const actualBtn = document.getElementById('actual-btn');
   const fileChosen = document.getElementById('file-chosen');
    
    actualBtn.addEventListener('change', function(){
      fileChosen.textContent = this.files[0].name
    });


   select.addEventListener('click',function(){

    if (select_list.style.display=="none"){
        select_list.style.display='block';
        select_head.classList.add('open');
    } else {
        select_list.style.display='none';
        select_head.classList.remove('open');
    }

   });



    ///////// slider adaptive
   let main_slide = document.getElementsByClassName('main')[0];
   let setHeight = null;
   if (window.screen.availWidth>1400){
        setHeight = (window.screen.availWidth * 656) / 1400;
        main_slide.style.minHeight = setHeight+"px";
    }
   
   window.addEventListener('resize',(e)=>{
    
    if (e.target.screen.availWidth>1400){
        setHeight = (e.target.screen.availWidth * 656) / 1400;
        main_slide.style.minHeight = setHeight+"px";
    }
    
   });


   //// video set
   let playbutton = document.getElementsByClassName('videoframe__playbutton')[0];
   let videoframe = document.getElementsByClassName('videoframe__video')[0];


   

   playbutton.addEventListener('click',(e)=>{
       e.preventDefault();
        playbutton.classList.add('activeframe');
        videoframe.setAttribute('src',videoframe.getAttribute('data-src'))
   });

    //////audio

   let audio = new Audio("../sc.mp3");
   let play = document.getElementsByClassName('audioplayer_button')[0];
   

   audio.addEventListener("loadeddata",() => {
      document.getElementsByClassName("audioplayer_time")[0].textContent = getTimeCodeFromNum(audio.duration);
      audio.volume = .9;
    },false);

    
    let audioplayer_triger = document.getElementsByClassName('audioplayer_timeline_progress_triger')[0];
    let audioplayer_progress = document.getElementsByClassName('audioplayer_timeline_progress')[0];
    let timeline = 0;
    let played = 0;
    const audioplayer_triger_start = ((audioplayer_triger.getBoundingClientRect()).left)+8;
    
    let updateTimeline = null;
    
    play.addEventListener('click',()=>{
        if (audio.paused) {
            play.classList.remove("audioplayer_play");
            play.classList.add("audioplayer_pause");
            audio.play();
            if (played==0){
                played==1;
                updateTimeline = setInterval(() => {
                    audioplayer_progress.style.width = audio.currentTime / audio.duration * 100 + "%";
                    audioplayer_triger.style.left = audio.currentTime / audio.duration * 100 + "%";
                    
                    document.getElementsByClassName("audioplayer_time")[0].textContent = getTimeCodeFromNum(audio.currentTime);
                    if(audio.currentTime==audio.duration){
                        play.classList.remove("audioplayer_pause");
                        play.classList.add("audioplayer_play");
                        audio.pause();
                        clearInterval(updateTimeline);
                        played=0;

                    }
                    console.log(`${audio.currentTime}//${audio.duration}`)
                }, 500);
            }
        } else {
            play.classList.remove("audioplayer_pause");
            play.classList.add("audioplayer_play");
            audio.pause();
            clearInterval(updateTimeline);
            played=0;
        }
    });
    
    
    audioplayer_triger.addEventListener('mousemove',(e)=>{
       
       if (timeline==1){
        setTimeline(e);
       }
    });

    audioplayer_triger.addEventListener('mousedown',(e)=>{
        timeline = 1;
    });
    audioplayer_triger.addEventListener('click',(e)=>{
        document.getElementsByClassName("audioplayer_time")[0].textContent = getTimeCodeFromNum(audio.currentTime);
        setTimeline(e);

    });

    audioplayer_triger.addEventListener('mouseup',(e)=>{
        timeline = 0;
    });

   
    window.addEventListener('mouseup',()=>{
        if(timeline==1){
            timeline=0;
        }
    })

    window.addEventListener('mousemove',(e)=>{
        if(timeline==1){
            document.getElementsByClassName("audioplayer_time")[0].textContent = getTimeCodeFromNum(audio.currentTime);
            setTimeline(e);
        }
    })
   
    function setTimeline(e){
        
        if (e.clientX>=audioplayer_triger_start && e.clientX<audioplayer_triger_start+119){
            if (e.clientX-audioplayer_triger_start>=1){
                audioplayer_progress.style.width = e.clientX-audioplayer_triger_start +'px';
                audioplayer_triger.style.left=e.clientX-audioplayer_triger_start +'px';
                setAudioDuration(e.clientX-audioplayer_triger_start);
            } else {
                audioplayer_progress.style.width = 0 +'px';
                audioplayer_triger.style.left=0 +'px';
                setAudioDuration(0);
            }
        }

    }

    function setAudioDuration(pixelWidth){
        if (pixelWidth==0){
            audio.currentTime=0;
            return null;
        }
        
        const val = ((pixelWidth * 100) / 119) * 0.01;
        audio.currentTime = val * audio.duration;
    }

    function getTimeCodeFromNum(num) {
        let seconds = parseInt(num);
        let minutes = parseInt(seconds / 60);
        seconds -= minutes * 60;
        const hours = parseInt(minutes / 60);
        minutes -= hours * 60;
      
        if (hours === 0) return `${minutes}:${String(seconds % 60).padStart(2, 0)}`;
        return `${String(hours).padStart(2, 0)}:${minutes}:${String(
          seconds % 60
        ).padStart(2, 0)}`;
      }


    ////// exemplpfication
    let phonewrapImg = document.getElementsByClassName('examplfication__phonewrap__phone')[0];
    let exemplpfication1 = document.getElementsByClassName('examplfication__item1')[0];
    let exemplpfication2 = document.getElementsByClassName('examplfication__item2')[0];
    let phonewrapStart = (((document.getElementsByClassName('onlineadvantage')[0]).getBoundingClientRect()).top);
    window.addEventListener('scroll',()=>{
        phonewrapStart = (((document.getElementsByClassName('onlineadvantage')[0]).getBoundingClientRect()).top);
        if(phonewrapStart<=200){
            phonewrapImg.classList.remove('examplfication__phonewrap__phone_onstart');
            setTimeout(()=>{
                exemplpfication1.classList.remove('examplfication__item1_onstart');
                setTimeout(()=>{
                    exemplpfication2.classList.remove('examplfication__item2_onstart');
                },800)
            },800)
        }
    });



});