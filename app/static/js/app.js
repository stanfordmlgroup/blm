/* Javascript for app lives here */



let dropArea = document.getElementById('drop-area')


;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false)
})

function preventDefaults (e) {
  e.preventDefault()
  e.stopPropagation()
}



;['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false)
})

;['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false)
})

function highlight(e) {
  dropArea.classList.add('highlight')
}

function unhighlight(e) {
  dropArea.classList.remove('highlight')
}

dropArea.addEventListener('drop', handleDrop, false)


function handleDrop(e) {
  let dt = e.dataTransfer
  let files = dt.files

  handleFiles(files)
}

$('#anonymize').hide();

function handleFiles(files) {
  files = [...files]
  // initializeProgress(1)
  file = files[0]
  // uploadFile(file)
  previewFile(file)
  $('#getfiles').hide();


  /* attach a submit handler to the form */
  $("#imagecollector").submit(
    function(event) {
    console.log('ASDSADSA');
    /* stop form from submitting normally */
    event.preventDefault();

    /* get the action attribute from the <form action=""> element */
    var $form = $(this),
      url = $form.attr('action');

    var formData = new FormData(this);
    /* Send the data using post with element id name and name2*/
    $.ajax({
          type: "post",
          url: url,
          data: formData,
          contentType: false,
          processData: false,
          cache: false,
          success: function (anon_data) {
              $('#statustext').text('Here is the anonymized image!');
              $('#statustext2').hide();
              $('#output').attr('src', 'data:image/jpeg;base64, ' + anon_data);
              $('#output').attr('width', '50%');
              $('#anonymize').show();
      
          },

          error: function(jqXHR, textStatus, errorThrown) {
              console.log(errorThrown);
          }
      })
  });


  $('#imagecollector').submit();
  $('#anonymize').show();
}



function uploadFile(file) {
  let url = 'YOUR URL HERE'
  let formData = new FormData()

  formData.append('file', file)

  fetch(url, {
    method: 'POST',
    body: formData
  })
  .then(progressDone) // <- Add `progressDone` call here
  .catch(() => { /* Error. Inform the user */ })
}



function previewFile(file) {
  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = function() {
    let img = document.createElement('img')
    img.src = reader.result
    if (document.getElementById('gallery').firstChild) {
        // It has at least one
        document.getElementById('gallery').replaceChild(img,document.getElementById('gallery').firstChild)
        // alert('Only one image can be uploaded');
    } else {
        document.getElementById('gallery').appendChild(img)
    }
  }
}

let filesDone = 0
let filesToDo = 0
let progressBar = document.getElementById('progress-bar')
function initializeProgress(numfiles) {
  progressBar.value = 0
  filesDone = 0
  filesToDo = numfiles
}

function progressDone() {
  filesDone++
  progressBar.value = filesDone / filesToDo * 100
}
