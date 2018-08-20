function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    if (typeof stroke == "undefined") {
        stroke = true;
    }
    if (typeof radius === "undefined") {
        radius = 5;
    }
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    if (stroke) {
        ctx.stroke();
    }
    if (fill) {
        ctx.fill();
    }
}

function Draw(faces, imgsize) {
    console.log(faces);
    var img = document.getElementById("uploaded-img");
    var cnvs = $('#img-canvas');

    cnvs[0].style.position = "absolute";
    cnvs[0].style.left = img.offsetLeft + "px";
    cnvs[0].style.top = img.offsetTop + "px";
    cnvs.attr("width", imgsize['width'] + "px");
    cnvs.attr("height", imgsize['height'] + "px");

    var ctx = cnvs[0].getContext("2d");

    console.log(faces[0]['top'], faces[0]['left'], faces[0]['width'], faces[0]['height']);
    faces.forEach(function (face) {
        ctx.beginPath();
        ctx.lineWidth = "1";
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#439ffe";
        ctx.fillStyle = "#439ffe";
        ctx.rect(face['top'], face['left'] - 20, face['width'], 20);
        ctx.fill();
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#ffffff";
        var rectHeight = 20;
        var rectWidth = face['width'];
        var rectX = face['top'];
        var rectY = face['left'] - 20;
        ctx.fillText(face['emotion'], rectX + (rectWidth / 2), rectY + (rectHeight / 2));
        ctx.strokeStyle = "#439ffe";
        ctx.rect(face['top'], face['left'], face['width'], face['height']);
        ctx.stroke();
    });
}

$(function () {
    $('#fileupload').fileupload({
        url: '/upload',
        dataType: 'json',
        add: function (e, data) {
            if ((/\.(gif|jpg|jpeg|tiff|png)$/i).test(data.files[0].name)) {
                if (data.files && data.files[0]) {
                    data.submit();
                }
            }
        },
        success: function (response, status) {
            console.log(response);
            uploadedImg = $('#uploaded-img');
            uploadedImg.attr("width", response['imgsize']['width'] + "px");
            uploadedImg.attr("height", response['imgsize']['height'] + "px");
            uploadedImg.attr("src", response['img']);
            Draw(response['faces'], response['imgsize']);

            var pretty = JSON.stringify(response, undefined, 4);
            document.getElementById('api-response-textarea').value = pretty;
            console.log(pretty);
        },
        error: function (error) {
            console.log(error);
        },
        progressall: function (e, data) {
            var progress = parseInt(data.loaded / data.total * 100, 10);
        }
    });
});