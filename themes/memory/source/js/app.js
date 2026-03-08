(function ($) {

  "use strict";

  var toggleActive = function (self, e) {
    e.preventDefault();
    if (self.hasClass("active") === true) {
      self.removeClass("active");
    } else {
      self.addClass("active");
    }
  };

  var switchSidebarTab = function (e) {
    var self = $(this),
      target = self.attr('data-toggle'),
      counter_target = target === 'toc' ? 'bio' : 'toc';
    if (self.hasClass('active')) {
      return;
    }
    toggleActive(self, e);
    toggleActive(self.siblings('.dark-btn'), e);
    $('.site-' + counter_target).toggleClass('show');
    setTimeout(function () {
      $('.site-' + counter_target).hide();
      $('.site-' + target).show();
      setTimeout(function () {
        $('.site-' + target).toggleClass('show');
      }, 50);
    }, 240);
  };

  var scrolltoElement = function (e) {
    e.preventDefault();
    var self = $(this),
      correction = e.data ? e.data.correction ? e.data.correction : 0 : 0;
    $('html, body').animate({'scrollTop': $(self.attr('href')).offset().top - correction}, 400);
  };

  var closeMenu = function (e) {
    e.stopPropagation();
    $('body').removeClass('menu-open');
    $('#site-nav-switch').removeClass('active');
  };

  var toggleMenu = function (e) {
    e.stopPropagation();
    $('body').toggleClass('menu-open');
    $('#site-nav-switch').toggleClass('active');
  };

  var pixivArchiveStat = function () {
    var vol = $(".article-entry ul").length;
    var artistCount = $(".article-entry ul li").length;
    $("#pixiv-vol").text(vol);
    $("#pixiv-artist-count").text(artistCount);
  };

  var initArticleShare = function () {
    var shareRoot = $(".article-share");
    if (!shareRoot.length) {
      return;
    }

    var wechatPanel = $(".wechat-share-panel");
    var wechatToggle = shareRoot.find(".share-wechat-toggle");
    var wechatClose = wechatPanel.find(".share-wechat-close");

    var setCopySuccess = function (btn) {
      var self = $(btn);
      var originalText = self.data("origin-text");
      if (!originalText) {
        originalText = self.text();
        self.data("origin-text", originalText);
      }
      self.text("已复制");
      self.addClass("is-success");
      setTimeout(function () {
        self.text(originalText);
        self.removeClass("is-success");
      }, 1400);
    };

    var fallbackCopyText = function (text, onSuccess) {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        if (document.execCommand("copy")) {
          onSuccess();
        }
      } catch (err) {
        console.error(err);
      }
      document.body.removeChild(textarea);
    };

    var copyText = function (text, onSuccess) {
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(onSuccess).catch(function () {
          fallbackCopyText(text, onSuccess);
        });
      } else {
        fallbackCopyText(text, onSuccess);
      }
    };

    shareRoot.add(wechatPanel).find(".share-copy").on("click", function () {
      var self = $(this);
      var shareUrl = self.attr("data-copy") || shareRoot.attr("data-share-url") || window.location.href;
      copyText(shareUrl, function () {
        setCopySuccess(self);
      });
    });

    wechatToggle.on("click", function () {
      var expanded = $(this).attr("aria-expanded") === "true";
      if (expanded) {
        wechatPanel.prop("hidden", true);
        $(this).attr("aria-expanded", "false");
      } else {
        wechatPanel.prop("hidden", false);
        $(this).attr("aria-expanded", "true");
      }
    });

    wechatClose.on("click", function () {
      wechatPanel.prop("hidden", true);
      wechatToggle.attr("aria-expanded", "false");
    });

    $(document).on("keydown", function (e) {
      if (e.key === "Escape" && !wechatPanel.prop("hidden")) {
        wechatPanel.prop("hidden", true);
        wechatToggle.attr("aria-expanded", "false");
      }
    });
  };

  $(function () {
    $('#footer, #main').addClass('loaded');
    $('#site-nav-switch').on('click', toggleMenu);
    $('#site-wrapper .overlay, #sidebar-close').on('click', closeMenu);
    $('.window-nav, .site-toc a').on('click', scrolltoElement);
    $(".content .video-container").fitVids();
    $('#site-sidebar .sidebar-switch .dark-btn').on('click', switchSidebarTab);

    if (window.location.pathname === '/pixiv' || window.location.pathname === '/pixiv/') {
      pixivArchiveStat();
    }

    initArticleShare();

    setTimeout(function () {
      $('#loading-bar-wrapper').fadeOut(500);
    }, 300);
  });

})(jQuery);
