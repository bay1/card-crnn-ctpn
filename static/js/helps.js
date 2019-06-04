function postImg() {
    //执行post请求，识别图片
    if (imgJson['num'] == 0) {
        loadingGif('loadingGif');
        imgJson['num'] = 1; //防止重复提交
        jQuery.ajax({
            type: "post",
            url: '/',
            data: JSON.stringify({
                "img": imgJson["imgString"]
            }),
            success: function (res) {
                loadingGif('loadingGif');
                imgJson['num'] = 0; //防止重复提交
                imgJson["result"] = res;
                getChildDetail();
            }
        });
    }

}


function loadingGif(loadingGif) {
    //加载请求时旋转动态图片
    var imgId = document.getElementById(loadingGif);
    if (imgId.style.display == "block") {
        imgId.style.display = "none";
    } else {
        imgId.style.display = "block";
    }
}


function resize_im(w, h, scale, max_scale) {
    f = parseFloat(scale) / Math.min(h, w);
    if (f * Math.max(h, w) > max_scale) {
        f = parseFloat(max_scale) / Math.max(h, w);
    }
    newW = parseInt(w * f);
    newH = parseInt(h * f);

    return [newW, newH, f]
}


function FunimgPreview(avatarSlect, avatarPreview) {
    //avatarSlect 上传文件控件
    //avatarPreview 预览图片控件
    jQuery("#" + avatarSlect).change(function () {
        var obj = jQuery("#" + avatarSlect)[0].files[0];

        var fr = new FileReader();
        fr.readAsDataURL(obj);
        fr.onload = function () {
            jQuery("#" + avatarPreview).attr('src', this.result);
            imgJson.imgString = this.result;
            var image = new Image();
            image.onload = function () {
                var width = image.width;
                var height = image.height;
                newWH = resize_im(width, height, 800, 1200);
                newW = newWH[0];
                newH = newWH[1];
                imgRate = newWH[2];
                imgJson.width = width;
                imgJson.height = height;
                jQuery("#" + avatarPreview).attr('width', newW);
                jQuery("#" + avatarPreview).attr('height', newH);
            };
            image.src = this.result;

            postImg(); //提交POST请求
        };

    })
}

function getChildDetail() {
    result = imgJson["result"];
    console.log(result);
    jQuery("#Preview").attr('src', 'data:image/png;base64,' + result['result_image']);
    showResult(result['text'], result['timeTake']);
}

//show数据
function showResult(text, timeTake) {
    jQuery(".show-result").empty();
    var p = "<span class=\"glyphicon glyphicon-time\" aria-hidden=\"true\"></span> <span class=\"text-warning\">" + timeTake + "s </span><br/>" +
        " <span class=\"glyphicon glyphicon-circle-arrow-right\" aria-hidden=\"true\"></span> <span class=\"text-warning\">" + text + "</span>";
    jQuery(".show-result").append(p);
}